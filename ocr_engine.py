"""
OCR Engine using Keras CRNN as primary model
Supports both printed text (EasyOCR) and handwritten text (Keras CRNN)
"""

import numpy as np
from PIL import Image
from typing import Any, Union
import os
import cv2

from config import EASYOCR_CONFIG, HANDWRITING_EASYOCR_CONFIG, MODEL_PATHS, CHAR_CLASSES
from preprocessing import (
    to_numpy, preprocess_image, enhance_for_ocr,
    upscale_for_handwriting
)
from postprocessing import post_process_handwriting, process_lines

# Type alias for image input
ImageInput = Union[Image.Image, np.ndarray[Any, Any]]


class OCREngine:
    """Main OCR Engine - Keras CRNN primary for handwritten, EasyOCR for printed"""
    
    def __init__(self) -> None:
        self._easyocr_reader = None
        self._handwriting_model = None
        self._char_model = None
        self.img_height, self.img_width, self.max_length = 32, 128, 32
        self.char_list = CHAR_CLASSES
        self.char_to_num = {char: idx for idx, char in enumerate(self.char_list)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.char_list)}
    
    @property
    def easyocr_reader(self):
        """Lazy-load EasyOCR reader"""
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')
        return self._easyocr_reader
    
    @property
    def handwriting_model(self):
        """Lazy-load Keras handwriting CRNN model"""
        if self._handwriting_model is None:
            self._handwriting_model = self.load_handwriting_model()
        return self._handwriting_model
    
    @property
    def char_model(self):
        """Lazy-load Keras character model"""
        if self._char_model is None:
            self._char_model = self.load_char_model()
        return self._char_model
    
    def load_handwriting_model(self):
        """Load Keras handwriting CRNN model from disk"""
        try:
            import keras
            model_path = MODEL_PATHS.get('handwriting_model', 'models/handwriting_model.keras')
            if os.path.exists(model_path):
                return keras.models.load_model(model_path)
            else:
                print(f"Warning: Handwriting model not found at {model_path}")
                return None
        except Exception as e:
            print(f"Error loading handwriting model: {e}")
            return None
    
    def load_char_model(self):
        """Load Keras character model from disk"""
        try:
            import keras
            model_path = MODEL_PATHS.get('char_model', 'models/char_model.keras')
            if os.path.exists(model_path):
                return keras.models.load_model(model_path)
            else:
                print(f"Warning: Character model not found at {model_path}")
                return None
        except Exception as e:
            print(f"Error loading character model: {e}")
            return None
    
    def recognize_text(self, image: ImageInput, mode: str = 'printed') -> dict[str, Any]:
        """
        Main OCR function
        
        Args:
            image: PIL Image or numpy array
            mode: 'printed' or 'handwritten'
        
        Returns:
            dict with text, confidence, lines
        """
        try:
            if mode == 'handwritten':
                return self._recognize_handwritten_keras(image)
            else:
                return self._recognize_printed(image)
        except Exception as e:
            return self._error_response(str(e))
    
    def _recognize_printed(self, image: ImageInput) -> dict[str, Any]:
        """Printed text recognition using EasyOCR"""
        try:
            base = preprocess_image(image)
            processed = enhance_for_ocr(base)
            return self._run_easyocr(processed, EASYOCR_CONFIG, 'printed')
        except Exception as e:
            return self._error_response(str(e))
    
    def _recognize_handwritten_keras(self, image: ImageInput) -> dict[str, Any]:
        """Handwritten text recognition using Keras CRNN model (PRIMARY)"""
        try:
            if self.handwriting_model is None:
                raise Exception("Handwriting model not loaded")
            
            # Preprocess for Keras CRNN
            img_array = self._preprocess_for_crnn(image)
            
            # Predict using CRNN
            predictions = self.handwriting_model.predict(img_array, verbose=0)
            
            # Decode predictions
            text, confidence = self._decode_crnn_predictions(predictions[0])
            
            # Split into lines if multiple lines detected
            lines = [text] if text else []
            
            return {
                'text': text,
                'confidence': float(confidence),
                'mode': 'handwritten',
                'word_boxes': [],
                'lines': lines,
                'line_count': len(lines)
            }
        except Exception as e:
            return self._error_response(str(e))
    
    def _preprocess_for_crnn(self, image: ImageInput) -> np.ndarray:
        """Preprocess image for CRNN model input"""
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Resize to CRNN input size
        resized = cv2.resize(img, (self.img_width, self.img_height))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add channel dimension
        img_proc = np.expand_dims(normalized, -1)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_proc, 0)
        
        return img_batch
    
    def _decode_crnn_predictions(self, predictions: np.ndarray) -> tuple[str, float]:
        """
        Decode CRNN predictions to text
        
        Args:
            predictions: Shape (max_length, num_classes)
        
        Returns:
            (text, confidence)
        """
        # Get character index for each position (argmax)
        char_indices = np.argmax(predictions, axis=1)
        
        # Get confidence scores (max probability)
        confidences = np.max(predictions, axis=1)
        
        # Convert indices to characters
        text_chars = []
        valid_confidences = []
        
        for idx, conf in zip(char_indices, confidences):
            # Skip padding (index 0)
            if idx == 0:
                continue
            
            # Map index to character
            if idx - 1 < len(self.char_list):
                char = self.char_list[idx - 1]
                text_chars.append(char)
                valid_confidences.append(float(conf))
        
        text = ''.join(text_chars)
        avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
        
        return text, avg_confidence
    
    def _run_easyocr(self, img_array: Any, config: dict[str, Any], mode: str) -> dict[str, Any]:
        """Run EasyOCR with given config"""
        results = self.easyocr_reader.readtext(
            img_array,
            batch_size=config.get('batch_size', 8),
            paragraph=config.get('paragraph', True),
            min_size=config.get('min_size', 10),
            decoder=config.get('decoder', 'beamsearch'),
            beamWidth=config.get('beamWidth', 5),
        )

        if not results:
            return self._empty_response(mode)

        word_boxes, confidences = self._parse_easyocr_results(results)
        lines = self._group_into_lines(word_boxes)
        combined_text = '\n'.join(lines)
        avg_confidence = np.mean(confidences) if confidences else 0

        # Clean up temporary fields
        for wb in word_boxes:
            wb.pop('y_center', None)
            wb.pop('x_left', None)

        return {
            'text': combined_text,
            'confidence': float(avg_confidence),
            'mode': mode,
            'word_boxes': word_boxes,
            'lines': lines,
            'line_count': len(lines)
        }
    
    def _parse_easyocr_results(self, results: list[Any]) -> tuple[list[dict[str, Any]], list[float]]:
        """Parse EasyOCR results into word boxes"""
        word_boxes = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            confidences.append(confidence)
            word_boxes.append({
                'text': text,
                'box': [[int(p[0]), int(p[1])] for p in bbox],
                'confidence': float(confidence),
                'y_center': (bbox[0][1] + bbox[2][1]) / 2,
                'x_left': bbox[0][0]
            })
        
        return word_boxes, confidences
    
    def _group_into_lines(self, word_boxes: list[dict[str, Any]]) -> list[str]:
        """Group word boxes into lines based on Y position"""
        if not word_boxes:
            return []
        
        sorted_boxes = sorted(word_boxes, key=lambda x: (x['y_center'], x['x_left']))
        
        # Calculate line threshold
        avg_height = np.mean([abs(wb['box'][2][1] - wb['box'][0][1]) for wb in word_boxes])
        line_threshold = avg_height * 0.8
        
        lines = []
        current_line = []
        last_y = -100
        
        for wb in sorted_boxes:
            y = wb['y_center']
            
            if abs(y - last_y) > line_threshold and current_line:
                line_text = ' '.join([w['text'] for w in sorted(current_line, key=lambda x: x['x_left'])])
                lines.append(line_text)
                current_line = []
            
            current_line.append(wb)
            last_y = y
        
        if current_line:
            line_text = ' '.join([w['text'] for w in sorted(current_line, key=lambda x: x['x_left'])])
            lines.append(line_text)
        
        return lines
    
    def _empty_response(self, mode: str) -> dict[str, Any]:
        """Empty OCR result"""
        return {
            'text': '',
            'confidence': 0,
            'mode': mode,
            'word_boxes': [],
            'lines': [],
            'line_count': 0
        }
    
    def _error_response(self, error: str) -> dict[str, Any]:
        """Error OCR result"""
        return {
            'success': False,
            'error': error,
            'text': '',
            'confidence': 0,
            'word_boxes': [],
            'lines': [],
            'line_count': 0
        }
