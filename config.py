"""
Configuration constants for OCR Engine
"""
from typing import Any

# EasyOCR Settings
EASYOCR_CONFIG: dict[str, Any] = {
    'languages': ['en'],
    'gpu': False,
    'batch_size': 4,
    'paragraph': False,
    'min_size': 10,
    'decoder': 'greedy',
    'beamWidth': 3,
}

# Handwriting EasyOCR Settings (more sensitive)
HANDWRITING_EASYOCR_CONFIG: dict[str, Any] = {
    'paragraph': False,
    'min_size': 5,
    'text_threshold': 0.4,
    'low_text': 0.3,
    'link_threshold': 0.2,
    'canvas_size': 3000,
    'mag_ratio': 2.0,
    'width_ths': 0.7,
    'height_ths': 0.7,
    'slope_ths': 0.3,
}

# Image Processing Settings
IMAGE_CONFIG: dict[str, Any] = {
    'max_width': 2000,
    'upscale_factor': 2.5,
    'upscale_threshold': 1500,
}

# Character classes for character model
CHAR_CLASSES = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

# Model paths
MODEL_PATHS = {
    'char_model': 'models/char_model.keras',
    'handwriting_model': 'models/handwriting_model.keras',
}
