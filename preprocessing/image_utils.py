"""Shared image processing utilities for OCR"""

import numpy as np
from typing import List, Tuple
from PIL import Image

ImageArray = np.ndarray


def otsu_threshold(image: np.ndarray) -> int:
    """
    Compute Otsu's automatic binarization threshold.
    Finds the threshold that maximizes between-class variance.
    
    Args:
        image: grayscale numpy array (uint8)
    
    Returns:
        int: threshold value (0-255)
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    
    pixel_count = image.size
    if pixel_count == 0:
        return 128
    
    sum_all = np.sum(np.arange(256) * hist)
    sum_bg = 0.0
    count_bg = 0
    max_variance = 0.0
    threshold = 0
    
    for t in range(256):
        count_bg += hist[t]
        count_fg = pixel_count - count_bg
        
        if count_bg == 0 or count_fg == 0:
            continue
        
        sum_bg += t * hist[t]
        sum_fg = sum_all - sum_bg
        
        mean_bg = sum_bg / count_bg
        mean_fg = sum_fg / count_fg
        variance = count_bg * count_fg * (mean_bg - mean_fg) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    return threshold


def resize_char(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize character crop to target dimensions, maintaining aspect ratio.
    Centers content on white canvas. Returns uint8 array.
    
    Args:
        image: input image array
        target_h: target height
        target_w: target width
    
    Returns:
        Resized uint8 array
    """
    if image.size == 0:
        return np.ones((target_h, target_w), dtype='uint8') * 255

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype('uint8')

    h, w = image.shape
    if h == 0 or w == 0:
        return np.ones((target_h, target_w), dtype='uint8') * 255

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    pil_img = Image.fromarray(image, mode='L')
    pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = np.ones((target_h, target_w), dtype='uint8') * 255
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = np.array(pil_resized)

    return canvas
