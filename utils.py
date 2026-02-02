"""
Backend utility functions
"""

import io
import base64
from typing import Any
from PIL import Image


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def encode_image_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def validate_image(image: Any) -> bool:
    """Validate image is readable"""
    try:
        if isinstance(image, Image.Image):
            image.verify()
            return True
        return False
    except Exception:
        return False


def get_image_info(image: Image.Image) -> dict[str, Any]:
    """Get basic image information"""
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format
    }
