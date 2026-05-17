"""Image preprocessing - convert images to standard format"""

import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_image(image) -> Image.Image:
    """Convert a PIL image or numpy array to a RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image.astype("uint8"), mode="L").convert("RGB")
        if image.ndim == 3:
            return Image.fromarray(image.astype("uint8")).convert("RGB")
    raise ValueError("Invalid image input")


