"""
Image preprocessing utilities for OCR
"""

import numpy as np
import cv2
from PIL import Image
from config import IMAGE_CONFIG


def to_numpy(image) -> np.ndarray:
    """Convert PIL Image to numpy array RGB"""
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    return image


def preprocess_image(image, target_size=None) -> np.ndarray:
    """
    Preprocess image for OCR
    - Convert to RGB
    - Downscale large images for faster processing
    """
    img_array = to_numpy(image)
    
    # Downscale very large images
    height, width = img_array.shape[:2]
    max_width = IMAGE_CONFIG['max_width']
    if width > max_width:
        scale = max_width / width
        new_size = (max_width, int(height * scale))
        img_array = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)
    
    if target_size:
        img_array = cv2.resize(img_array, target_size)
    
    return img_array


def enhance_for_ocr(image) -> np.ndarray:
    """
    Enhanced preprocessing for better OCR accuracy
    - Increase contrast using CLAHE
    - Sharpen text
    """
    img_array = to_numpy(image)
    
    # Convert to LAB for better contrast enhancement
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend 50/50
    return cv2.addWeighted(enhanced, 0.5, sharpened, 0.5, 0)


def upscale_for_handwriting(image, scale_factor=None) -> np.ndarray:
    """Upscale small images for better handwriting detection"""
    img_array = to_numpy(image)
    scale = scale_factor or IMAGE_CONFIG['upscale_factor']
    
    height, width = img_array.shape[:2]
    if width < IMAGE_CONFIG['upscale_threshold']:
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(img_array, new_size, interpolation=cv2.INTER_CUBIC)
    
    return img_array


def binarize_handwriting(image) -> np.ndarray:
    """Binarize image for handwriting - black text on white background"""
    img_array = to_numpy(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure dark text on light background
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def preprocess_handwriting_advanced(image) -> np.ndarray:
    """
    Advanced preprocessing for handwriting:
    - Upscale, denoise, enhance contrast, morphological operations
    """
    img_array = to_numpy(image)
    
    # Upscale
    height, width = img_array.shape[:2]
    if width < 2000:
        scale = 3.0
        img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    result = cv2.erode(dilated, kernel, iterations=1)
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def to_grayscale_enhanced(image) -> np.ndarray:
    """Convert to grayscale with CLAHE enhancement"""
    img_array = to_numpy(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
