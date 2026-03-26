"""
Configuration constants for OCR Engine
"""
from typing import Any

# Image Processing Settings
IMAGE_CONFIG: dict[str, Any] = {
    'max_width': 2000,
    'upscale_factor': 2.5,
    'upscale_threshold': 1500,
}

# Character classes for character model
# MUST match the training data structure in train_on_real_emnist.py
# Model trained on EMNIST byclass: 0-9 (digits) + A-Z (uppercase letters) = 36 classes
CHAR_CLASSES = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Model paths
MODEL_PATHS = {
    'char_model': 'models/char_model.keras',
    'handwriting_model': 'models/handwriting_model.keras',
}
