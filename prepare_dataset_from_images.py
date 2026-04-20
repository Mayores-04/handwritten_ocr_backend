"""
Prepare handwriting training dataset from raw images

This script helps you convert raw photos/scans of handwritten text into a 
properly formatted dataset for the CRNN training script.

Usage:
    1. Place your handwritten photo/scan images in a folder (e.g., raw_images/)
    2. Run: python prepare_dataset_from_images.py --input-dir ./raw_images --output-dir ./data
    3. Manually edit data/labels.txt with transcriptions
    4. Run: python train_on_real_handwriting.py

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def auto_crop_handwriting(image: np.ndarray) -> np.ndarray:
    """
    Auto-crop image to remove white borders around handwritten text
    
    Args:
        image: Grayscale image
        
    Returns:
        Cropped image containing handwritten text
    """
    # Find non-white pixels (handwriting)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find text
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get bounding box of all contours
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add small margin
    margin = 5
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    
    if x_min >= x_max or y_min >= y_max:
        return image
    
    return image[y_min:y_max, x_min:x_max]


def preprocess_image(image_path: str, target_height: int = 32, max_width: int = 512) -> Tuple[np.ndarray, bool]:
    """
    Read and preprocess image for training
    
    Args:
        image_path: Path to image file
        target_height: Target height (32 pixels standard)
        max_width: Maximum width before truncation
        
    Returns:
        (Processed image, success flag)
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to read: {image_path}")
            return None, False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Auto-crop white space
        gray = auto_crop_handwriting(gray)
        
        # Resize maintaining aspect ratio
        h, w = gray.shape
        aspect = w / h
        new_h = target_height
        new_w = int(target_height * aspect)
        
        # Resize
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to fixed width with white (255)
        padded = np.ones((target_height, max_width), dtype=np.uint8) * 255
        padded[:, :min(new_w, max_width)] = resized[:, :min(new_w, max_width)]
        
        # Normalize to [0, 1]
        normalized = padded.astype(np.float32) / 255.0
        
        return normalized, True
        
    except Exception as e:
        logger.warning(f"Error processing {image_path}: {e}")
        return None, False


def prepare_dataset(input_dir: str, output_dir: str, val_split: float = 0.2):
    """
    Prepare dataset from raw images
    
    Args:
        input_dir: Directory containing raw handwritten images
        output_dir: Output directory for processed dataset
        val_split: Validation split ratio (default 0.2 = 80% train, 20% val)
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("PREPARING HANDWRITING DATASET")
    logger.info("="*70)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        logger.error("No image files found in input directory")
        logger.error(f"Supported formats: {', '.join(image_extensions)}")
        return False
    
    image_files = sorted(image_files)
    logger.info(f"\nFound {len(image_files)} images")
    
    # Split into train/val
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    logger.info(f"Train: {len(train_files)} | Val: {len(val_files)}")
    
    # Process images
    processed_count = 0
    labels = []
    
    logger.info(f"\n[Processing] Training set ({len(train_files)} images)")
    for i, img_path in enumerate(train_files):
        processed_img, success = preprocess_image(str(img_path))
        if success:
            # Save processed image
            output_filename = f"line_{i:04d}.png"
            output_path_full = train_dir / output_filename
            
            # Convert from [0, 1] to [0, 255] for saving
            output_img = (processed_img * 255).astype(np.uint8)
            cv2.imwrite(str(output_path_full), output_img)
            
            labels.append((output_filename, "[TRANSCRIPTION NEEDED]"))
            processed_count += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed: {i + 1}/{len(train_files)}")
    
    logger.info(f"\n[Processing] Validation set ({len(val_files)} images)")
    for i, img_path in enumerate(val_files):
        processed_img, success = preprocess_image(str(img_path))
        if success:
            output_filename = f"line_{split_idx + i:04d}.png"
            output_path_full = val_dir / output_filename
            
            output_img = (processed_img * 255).astype(np.uint8)
            cv2.imwrite(str(output_path_full), output_img)
            
            labels.append((output_filename, "[TRANSCRIPTION NEEDED]"))
            processed_count += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed: {i + 1}/{len(val_files)}")
    
    # Create labels.txt
    logger.info(f"\n[Creating] labels.txt")
    labels_file = output_path / "labels.txt"
    with open(labels_file, 'w', encoding='utf-8') as f:
        f.write("# Handwritten Text Labels\n")
        f.write("# Format: filename,transcription\n")
        f.write("# Edit the transcriptions below with the actual text\n\n")
        for filename, placeholder in sorted(labels):
            f.write(f"{filename},{placeholder}\n")
    
    logger.info(f"\n✓ Dataset prepared successfully!")
    logger.info(f"✓ Processed {processed_count}/{len(image_files)} images")
    logger.info(f"✓ Saved to: {output_dir}/")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS:")
    logger.info("="*70)
    logger.info(f"1. Edit {output_dir}/labels.txt")
    logger.info(f"   Replace '[TRANSCRIPTION NEEDED]' with actual text")
    logger.info(f"   \n   Example:")
    logger.info(f"   line_0000.png,The quick brown fox jumps over the lazy dog")
    logger.info(f"   line_0001.png,Hello world this is handwriting")
    logger.info(f"\n2. Train the CRNN model:")
    logger.info(f"   python train_on_real_handwriting.py --dataset-path ./data")
    logger.info("="*70)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare handwriting dataset from raw images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare images from a folder
  python prepare_dataset_from_images.py --input-dir ./raw_images --output-dir ./data
  
  # Change validation split (30% validation, 70% training)
  python prepare_dataset_from_images.py --input-dir ./raw_images --val-split 0.3
        """
    )
    
    parser.add_argument(
        "--input-dir",
        default="./raw_images",
        help="Input directory with raw handwritten images (default: ./raw_images)"
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory for processed dataset (default: ./data)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2 = 80% train, 20% val)"
    )
    
    args = parser.parse_args()
    
    success = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_split=args.val_split
    )
    
    sys.exit(0 if success else 1)
