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

# Character classes for models/char_model.keras.
# The saved model has 62 output units: digits + uppercase + lowercase.
CHAR_CLASSES = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

# Model paths
MODEL_PATHS = {
    'char_model': 'models/char_model.keras',
    'handwriting_model': 'models/handwriting_model.keras',
}
