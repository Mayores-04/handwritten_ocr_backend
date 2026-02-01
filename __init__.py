"""
OCR Backend Package
"""

from .ocr_engine import OCREngine, create_handwriting_model
from .models import model_loader
from .config import EASYOCR_CONFIG, HANDWRITING_EASYOCR_CONFIG, CHAR_CLASSES

__all__ = [
    'OCREngine',
    'create_handwriting_model',
    'model_loader',
    'EASYOCR_CONFIG',
    'HANDWRITING_EASYOCR_CONFIG',
    'CHAR_CLASSES'
]
