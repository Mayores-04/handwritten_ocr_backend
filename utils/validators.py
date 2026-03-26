"""Input validation utilities"""

from typing import Any
from PIL import Image
import numpy as np


def validate_image(image: Any) -> tuple[bool, str]:
    """
    Validate image is readable and suitable for OCR
    Returns: (is_valid, error_message)
    """
    if image is None:
        return False, "No image provided"
    
    if not isinstance(image, Image.Image):
        return False, "Image must be a PIL Image object"
    
    try:
        # Check image properties
        if image.width < 10 or image.height < 10:
            return False, "Image too small (minimum 10x10 pixels)"
        
        # No upper limit - we handle any size
        # if image.width > 4000 or image.height > 4000:
        #     return False, "Image too large (maximum 4000x4000 pixels)"
        
        # Check if image is valid
        image.tobytes()
        return True, ""
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def validate_mode(mode: str) -> tuple[bool, str]:
    """Validate OCR mode"""
    valid_modes = ['printed', 'handwritten']
    if mode not in valid_modes:
        return False, f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
    return True, ""
