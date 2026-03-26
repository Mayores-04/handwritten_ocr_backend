"""General utility helper functions for OCR processing"""

import io
import base64
from typing import Any
from PIL import Image


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def encode_image_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Encode PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image to base64: {str(e)}")


def get_image_info(image: Image.Image) -> dict[str, Any]:
    """Get basic image information"""
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
        'size_mb': (image.width * image.height * 4) / (1024 * 1024)
    }
