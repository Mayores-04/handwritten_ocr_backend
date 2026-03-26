"""Image preprocessing - convert images to standard format"""

import numpy as np
import logging
from PIL import Image
from typing import List, Tuple

logger = logging.getLogger(__name__)


def preprocess_image(image) -> Image.Image:
    """Convert image to PIL Image with RGB mode"""
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    raise ValueError("Invalid image input")


