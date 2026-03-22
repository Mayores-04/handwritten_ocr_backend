"""
OCR Engine using EasyOCR for Image to Text Recognition
Optimized for free-tier deployment (512MB RAM limit)
Supports both printed text and handwritten text
"""

import numpy as np
from PIL import Image
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
    """Main OCR Engine - EasyOCR only for free-tier compatibility"""
    
    def __init__(self) -> None:
        self._easyocr_reader = None
    
    @property
    def easyocr_reader(self):
        """Lazy-load EasyOCR reader"""
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')
        return self._easyocr_reader
    
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
                return self._recognize_handwritten(image)
            else:
                return self._recognize_printed(image)
        except Exception as e:
            return self._error_response(str(e))
    
    def _recognize_printed(self, image: ImageInput) -> dict[str, Any]:
        """Printed text recognition"""
        try:
            base = preprocess_image(image)
            processed = enhance_for_ocr(base)
            return self._run_easyocr(processed, EASYOCR_CONFIG, 'printed')
        except Exception as e:
            return self._error_response(str(e))
    
    def _recognize_handwritten(self, image: ImageInput) -> dict[str, Any]:
        """Handwritten text recognition"""
        try:
            base = preprocess_image(image)
            processed = upscale_for_handwriting(base, 2.0)
            result = self._run_easyocr(processed, HANDWRITING_EASYOCR_CONFIG, 'handwritten')
            
            if result.get('text'):
                result['text'] = post_process_handwriting(result['text'])
                if result.get('lines'):
                    result['lines'] = process_lines(result['lines'])
            
            return result
        except Exception as e:
            return self._error_response(str(e))
    
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
