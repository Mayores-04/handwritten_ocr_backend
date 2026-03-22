"""
OCR Engine using EasyOCR for Image to Text Recognition
Supports both printed text and handwritten text
Note: Keras/TensorFlow removed for free-tier deployment (512MB RAM limit)
"""

import numpy as np
from PIL import Image
import cv2
from typing import Any, Union

from config import EASYOCR_CONFIG, HANDWRITING_EASYOCR_CONFIG
from preprocessing import (
    to_numpy, preprocess_image, enhance_for_ocr,
    upscale_for_handwriting
)
from postprocessing import post_process_handwriting, process_lines

# Type alias for image input
ImageInput = Union[Image.Image, np.ndarray[Any, Any]]


class OCREngine:
    """Main OCR Engine class - EasyOCR only"""
    
    def __init__(self) -> None:
        self._easyocr_reader = None
        self._handwriting_reader = None
    
    @property
    def easyocr_reader(self):
        """Lazy-load EasyOCR reader for printed text"""
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')
        return self._easyocr_reader
    
    @property
    def handwriting_reader(self):
        """Lazy-load EasyOCR reader for handwriting"""
        if self._handwriting_reader is None:
            import easyocr
            self._handwriting_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')
        return self._handwriting_reader
    
    def recognize_text(self, image: ImageInput, mode: str = 'printed') -> dict[str, Any]:
        """
        Main OCR function using EasyOCR
        
        Args:
            image: PIL Image or numpy array
            mode: 'printed' or 'handwritten'
        
        Returns:
            dict with text, confidence, and word boxes
        """
        try:
            if mode == 'handwritten':
                return self._recognize_handwritten(image)
            else:
                return self._recognize_printed(image)
        except Exception as e:
            return self._error_response(str(e))
    
    def _recognize_handwritten(self, image: ImageInput) -> dict[str, Any]:
        """Handwritten text recognition using EasyOCR"""
        try:
            base = preprocess_image(image)
            processed = upscale_for_handwriting(base, 2.0)
            result = self._run_easyocr(processed, HANDWRITING_EASYOCR_CONFIG)
            result['mode'] = 'handwritten'
            
            if result.get('text'):
                result['text'] = post_process_handwriting(result['text'])
                if result.get('lines'):
                    result['lines'] = process_lines(result['lines'])
            
            return result
        except Exception as e:
            return self._error_response(str(e))
    
    # ============ Printed Text Recognition ============
    
    def _recognize_printed(self, image: ImageInput) -> dict[str, Any]:
        """Single-pass printed OCR - fast and effective"""
        try:
            base = preprocess_image(image)
            # Single best preprocessing: enhance for better text detection
            processed = enhance_for_ocr(base)
            result = self._run_easyocr_printed(processed, 'easyocr')
            result['mode'] = 'easyocr'
            return result
        except Exception as e:
            return self._error_response(str(e))

    def _run_easyocr_printed(self, img_array: Any, mode_tag: str) -> dict[str, Any]:
        """Run EasyOCR and format output"""
        results = self.easyocr_reader.readtext(
            img_array,
            batch_size=EASYOCR_CONFIG['batch_size'],
            paragraph=EASYOCR_CONFIG['paragraph'],
            min_size=EASYOCR_CONFIG['min_size'],
            decoder=EASYOCR_CONFIG['decoder'],
            beamWidth=EASYOCR_CONFIG['beamWidth'],
        )

        if not results:
            return self._empty_response(mode_tag)

        word_boxes, confidences = self._parse_easyocr_results(results)
        lines = self._group_into_lines(word_boxes)
        combined_text = '\n'.join(lines)
        avg_confidence = np.mean(confidences) if confidences else 0

        for wb in word_boxes:
            wb.pop('y_center', None)
            wb.pop('x_left', None)

        return {
            'text': combined_text,
            'confidence': float(avg_confidence),
            'mode': mode_tag,
            'word_boxes': word_boxes,
            'lines': lines,
            'line_count': len(lines)
        }
    
    def _run_easyocr(self, img_array: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Run EasyOCR with given config and format output"""
        reader = self.handwriting_reader if config == HANDWRITING_EASYOCR_CONFIG else self.easyocr_reader
        
        results = reader.readtext(
            img_array,
            batch_size=config.get('batch_size', 8),
            paragraph=config.get('paragraph', True),
            min_size=config.get('min_size', 10),
            decoder=config.get('decoder', 'beamsearch'),
            beamWidth=config.get('beamWidth', 5),
        )

        if not results:
            return self._empty_response('easyocr')

        word_boxes, confidences = self._parse_easyocr_results(results)
        lines = self._group_into_lines(word_boxes)
        combined_text = '\n'.join(lines)
        avg_confidence = np.mean(confidences) if confidences else 0

        for wb in word_boxes:
            wb.pop('y_center', None)
            wb.pop('x_left', None)

        return {
            'text': combined_text,
            'confidence': float(avg_confidence),
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
                'confidence': confidence,
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
    
    # ============ Handwriting Recognition ============
    
    def _easyocr_handwriting(self, img_array: Any) -> dict[str, Any]:
        """Run EasyOCR with handwriting-optimized parameters"""
        try:
            results = self.easyocr_reader.readtext(
                img_array,
                detail=1,
                paragraph=HANDWRITING_EASYOCR_CONFIG['paragraph'],
                min_size=HANDWRITING_EASYOCR_CONFIG['min_size'],
                text_threshold=HANDWRITING_EASYOCR_CONFIG['text_threshold'],
                low_text=HANDWRITING_EASYOCR_CONFIG['low_text'],
                link_threshold=HANDWRITING_EASYOCR_CONFIG['link_threshold'],
                canvas_size=HANDWRITING_EASYOCR_CONFIG['canvas_size'],
                mag_ratio=HANDWRITING_EASYOCR_CONFIG['mag_ratio'],
                width_ths=HANDWRITING_EASYOCR_CONFIG['width_ths'],
                height_ths=HANDWRITING_EASYOCR_CONFIG['height_ths'],
                slope_ths=HANDWRITING_EASYOCR_CONFIG['slope_ths'],
            )
            
            if not results:
                return self._empty_response('easyocr_handwriting')
            
            word_boxes, confidences = self._parse_handwriting_results(results)
            lines = self._group_handwriting_lines(word_boxes)
            
            return {
                'text': '\n'.join(lines),
                'confidence': float(np.mean(confidences)) if confidences else 0,
                'lines': lines,
                'line_count': len(lines)
            }
        except Exception as e:
            return self._error_response(str(e))
    
    def _parse_handwriting_results(self, results: list[Any]) -> tuple[list[dict[str, Any]], list[float]]:
        """Parse EasyOCR results for handwriting"""
        word_boxes = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            confidences.append(confidence)
            y_top = min(bbox[0][1], bbox[1][1])
            y_bottom = max(bbox[2][1], bbox[3][1])
            word_boxes.append({
                'text': text,
                'y_center': (y_top + y_bottom) / 2,
                'y_top': y_top,
                'y_bottom': y_bottom,
                'x_left': min(bbox[0][0], bbox[3][0]),
                'height': abs(y_bottom - y_top)
            })
        
        return word_boxes, confidences
    
    def _group_handwriting_lines(self, word_boxes: list[dict[str, Any]]) -> list[str]:
        """Group handwriting words into lines with overlap detection"""
        if not word_boxes:
            return []
        
        sorted_boxes = sorted(word_boxes, key=lambda x: (x['y_center'], x['x_left']))
        
        heights = [wb['height'] for wb in word_boxes]
        line_threshold = np.median(heights) * 0.6 if heights else 30
        
        lines = []
        current_line = []
        current_y_range = None
        
        for wb in sorted_boxes:
            y = wb['y_center']
            
            if current_y_range is None:
                current_line = [wb]
                current_y_range = (wb['y_top'], wb['y_bottom'])
            else:
                line_y = (current_y_range[0] + current_y_range[1]) / 2
                
                if abs(y - line_y) <= line_threshold:
                    current_line.append(wb)
                    current_y_range = (
                        min(current_y_range[0], wb['y_top']),
                        max(current_y_range[1], wb['y_bottom'])
                    )
                else:
                    line_text = ' '.join([w['text'] for w in sorted(current_line, key=lambda x: x['x_left'])])
                    lines.append(line_text)
                    current_line = [wb]
                    current_y_range = (wb['y_top'], wb['y_bottom'])
        
        if current_line:
            line_text = ' '.join([w['text'] for w in sorted(current_line, key=lambda x: x['x_left'])])
            lines.append(line_text)
        
        return lines

    def _pick_best_result(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Choose the best OCR result using confidence and text quality heuristics"""
        def score(item: dict[str, Any]) -> float:
            confidence = float(item.get('confidence') or 0.0)
            text = str(item.get('text') or '')
            alnum_count = sum(ch.isalnum() for ch in text)
            text_len = len(text.strip())
            density = (alnum_count / text_len) if text_len else 0.0
            symbol_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
            symbol_ratio = (symbol_count / text_len) if text_len else 1.0
            repeated_char_runs = sum(1 for i in range(2, len(text)) if text[i] == text[i - 1] == text[i - 2])

            return (
                (confidence * 0.72)
                + (min(text_len, 220) / 220 * 0.20)
                + (density * 0.12)
                - (symbol_ratio * 0.08)
                - (min(repeated_char_runs, 8) * 0.01)
            )

        return max(results, key=score)
    
    # ============ TrOCR Recognition ============
    
    def _trocr_recognize(self, image: ImageInput) -> dict[str, Any]:
        """Use TrOCR for handwriting recognition"""
        try:
            pil_image = self._to_pil(image)
            line_images = self._detect_text_lines(pil_image)
            
            if not line_images:
                line_images = [pil_image]
            
            lines = []
            confidences = []
            
            for line_img in line_images:
                pixel_values = model_loader.trocr_processor(images=line_img, return_tensors="pt").pixel_values
                generated_ids = model_loader.trocr_model.generate(pixel_values, max_length=128)
                text = model_loader.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if text.strip():
                    lines.append(text.strip())
                    confidences.append(0.85)
            
            return {
                'text': '\n'.join(lines),
                'confidence': float(np.mean(confidences)) if confidences else 0,
                'mode': 'trocr_handwriting',
                'lines': lines,
                'line_count': len(lines)
            }
        except Exception as e:
            return self._handwriting_fallback(image)
    
    def _detect_text_lines(self, pil_image: Image.Image) -> list[Image.Image]:
        """Detect text lines using horizontal projection"""
        img_array = np.array(pil_image.convert('L'))
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h_projection = np.sum(binary, axis=1)
        threshold = np.max(h_projection) * 0.05
        text_rows = h_projection > threshold
        
        lines = []
        in_line = False
        start = 0
        
        for i, has_text in enumerate(text_rows):
            if has_text and not in_line:
                start = max(0, i - 5)
                in_line = True
            elif not has_text and in_line:
                end = min(len(text_rows), i + 5)
                if end - start > 10:
                    lines.append((start, end))
                in_line = False
        
        if in_line:
            lines.append((start, len(text_rows)))
        
        return [pil_image.crop((0, s, pil_image.width, e)) for s, e in lines]
    
    # ============ Keras Model Recognition ============
    
    def _keras_handwriting_recognize(self, image: ImageInput) -> dict[str, Any]:
        """
        Use custom Keras CRNN model for handwriting recognition
        Processes image through trained Keras model
        """
        if not self.handwriting_model:
            return self._error_response('Keras handwriting model not loaded')
        
        try:
            # Preprocess image for Keras
            img_array = self._preprocess_for_keras(image)
            
            # Get predictions from Keras model
            predictions = self.handwriting_model.predict(img_array, verbose=0)
            
            # Decode predictions
            text = self._ctc_decode(predictions)
            
            # Calculate confidence from predictions
            confidence = float(np.max(predictions)) if predictions.size > 0 else 0.5
            
            if not text or len(text.strip()) == 0:
                return self._error_response('Keras model produced empty output')
            
            return {
                'text': text,
                'confidence': confidence,
                'lines': [text],
                'line_count': 1,
                'mode': 'keras_handwriting'
            }
        except Exception as e:
            return self._error_response(f'Keras inference failed: {str(e)}')
    
    def _preprocess_for_keras(self, image: ImageInput) -> np.ndarray[Any, Any]:
        """
        Preprocess image for Keras CRNN model (32x128 grayscale)
        Input: PIL Image or numpy array
        Output: Normalized grayscale (1, 32, 128, 1)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert('RGB'))
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
        
        # Apply denoising and binarization
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to maintain text connectivity
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        # Resize to model input size (32x128)
        gray = cv2.resize(gray, (128, 32))
        
        # Normalize to [0, 1]
        gray = gray.astype('float32') / 255.0
        
        # Add batch and channel dimensions: (batch=1, height=32, width=128, channels=1)
        return np.expand_dims(np.expand_dims(gray, axis=-1), axis=0)
    
    def _ctc_decode(self, predictions: Any) -> str:
        """
        Decode CTC predictions from Keras CRNN model
        Removes blanks and consecutive duplicates
        """
        if predictions.size == 0:
            return ""
        
        # Get the character with highest probability at each timestep
        decoded = []
        prev_char_idx = 0
        
        for timestep in predictions[0]:
            char_idx = np.argmax(timestep)
            
            # Skip blank (CTC blank = index 0)
            # Skip consecutive duplicates
            if char_idx != 0 and char_idx != prev_char_idx:
                if char_idx - 1 < len(self.char_list):
                    decoded.append(self.char_list[char_idx - 1])
            
            prev_char_idx = char_idx
        
        return ''.join(decoded)
    
    # ============ Utility Methods ============
    
    def _to_pil(self, image: ImageInput) -> Image.Image:
        """Convert image to PIL Image"""
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    def _empty_response(self, mode: str) -> dict[str, Any]:
        return {
            'text': '',
            'confidence': 0,
            'mode': mode,
            'word_boxes': [],
            'lines': [],
            'line_count': 0
        }
    
    def _error_response(self, error: str) -> dict[str, Any]:
        return {
            'text': '',
            'confidence': 0,
            'mode': 'error',
            'error': error
        }


# ============ Model Creation (for training) ============

def create_handwriting_model(input_shape=(32, 128, 1), num_classes=63):
    """Create a CRNN model for handwriting recognition training"""
    from tensorflow import keras
    from tensorflow.keras import layers
    
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # CNN Feature Extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # Reshape for RNN
    new_shape = (x.shape[2], x.shape[1] * x.shape[3])
    x = layers.Reshape(target_shape=new_shape)(x)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    
    outputs = layers.Dense(num_classes + 1, activation='softmax', name='output')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='handwriting_crnn')


if __name__ == '__main__':
    model = create_handwriting_model()
    model.summary()
    print(f"\nHandwriting CRNN model created! Parameters: {model.count_params():,}")
