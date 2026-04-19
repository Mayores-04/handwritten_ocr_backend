"""
Convert IAM Handwriting Database to CRNN training format

Converts the IAM dataset structure:
  archive/iam_words/words/a01/a01-000u/a01-000u-00-00.png
  
Into training format:
  data/train/word_0001.png
  data/labels.txt (with transcriptions)
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_iam_metadata(metadata_file: str) -> Dict[str, str]:
    """
    Parse IAM words.txt metadata file
    
    Format of words.txt:
        a01-000u-00-00 ok 154 408 768 27 51 AT A
        where last field is transcription
    """
    
    metadata = {}
    
    logger.info(f"Parsing metadata from {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            filename = parts[0]  # e.g., "a01-000u-00-00"
            transcription = parts[-1]  # Last field is transcription
            
            # Skip error entries
            if parts[1] == 'err':
                continue
            
            metadata[filename] = transcription
    
    logger.info(f"Loaded {len(metadata)} word labels")
    return metadata


def find_iam_images(iam_root: str, metadata: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    Find all IAM word images and match with transcriptions
    
    Returns:
        {filename: (image_path, transcription), ...}
    """
    
    logger.info("Searching for IAM word images...")
    
    images = {}
    iam_path = Path(iam_root)
    words_dir = iam_path / "words"
    
    if not words_dir.exists():
        logger.error(f"Words directory not found: {words_dir}")
        return {}
    
    # Recursively find all .png files
    for png_file in words_dir.glob("**/*.png"):
        # Get the base filename without extension
        base_name = png_file.stem  # e.g., "a01-000u-00-00"
        
        if base_name in metadata:
            transcription = metadata[base_name]
            images[base_name] = (str(png_file), transcription)
    
    logger.info(f"Found {len(images)} word images with labels")
    return images


def convert_iam_dataset(
    iam_archive_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    max_samples: int = None
) -> bool:
    """
    Convert IAM dataset to CRNN training format
    
    Args:
        iam_archive_dir: Path to archive/iam_words directory
        output_dir: Output directory (will create train/, val/, labels.txt)
        val_split: Validation split ratio
        max_samples: Max samples to use (None = all)
    """
    
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Converting IAM Dataset to CRNN Format")
    logger.info("="*70)
    
    # Parse metadata
    metadata_file = Path(iam_archive_dir) / "words.txt"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False
    
    metadata = parse_iam_metadata(str(metadata_file))
    
    # Find images
    images = find_iam_images(iam_archive_dir, metadata)
    
    if not images:
        logger.error("No images found!")
        return False
    
    # Limit samples if requested
    if max_samples:
        image_items = list(images.items())[:max_samples]
        images = {k: v for k, v in image_items}
        logger.info(f"Using first {len(images)} samples")
    
    # Split into train/val
    image_list = list(images.items())
    np.random.shuffle(image_list)
    split_idx = int(len(image_list) * (1 - val_split))
    
    train_images = image_list[:split_idx]
    val_images = image_list[split_idx:]
    
    logger.info(f"\nTrain: {len(train_images)} | Val: {len(val_images)}")
    
    # Copy images and collect labels
    labels = []
    
    logger.info("\n[Copying] Training images...")
    for idx, (filename, (image_path, transcription)) in enumerate(train_images):
        try:
            output_filename = f"word_{idx:05d}.png"
            output_path_full = train_dir / output_filename
            
            # Copy image
            shutil.copy2(image_path, output_path_full)
            
            labels.append((output_filename, transcription))
            
            if (idx + 1) % 500 == 0:
                logger.info(f"  Copied: {idx + 1}/{len(train_images)}")
        except Exception as e:
            logger.warning(f"Error copying {image_path}: {e}")
    
    logger.info(f"✓ Copied {len(train_images)} training images")
    
    logger.info("\n[Copying] Validation images...")
    for idx, (filename, (image_path, transcription)) in enumerate(val_images):
        try:
            output_filename = f"word_{split_idx + idx:05d}.png"
            output_path_full = val_dir / output_filename
            
            # Copy image
            shutil.copy2(image_path, output_path_full)
            
            labels.append((output_filename, transcription))
            
            if (idx + 1) % 500 == 0:
                logger.info(f"  Copied: {idx + 1}/{len(val_images)}")
        except Exception as e:
            logger.warning(f"Error copying {image_path}: {e}")
    
    logger.info(f"✓ Copied {len(val_images)} validation images")
    
    # Create labels.txt
    logger.info(f"\n[Creating] labels.txt...")
    labels_file = output_path / "labels.txt"
    with open(labels_file, 'w', encoding='utf-8') as f:
        f.write("# IAM Handwriting Database - Converted for CRNN Training\n")
        f.write("# Format: filename,transcription\n\n")
        for filename, transcription in sorted(labels):
            f.write(f"{filename},{transcription}\n")
    
    logger.info(f"✓ Created labels.txt with {len(labels)} entries")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Conversion Complete!")
    logger.info("="*70)
    logger.info(f"✓ Train images: {len(train_images)} → {train_dir}")
    logger.info(f"✓ Val images: {len(val_images)} → {val_dir}")
    logger.info(f"✓ Labels: {labels_file}")
    
    logger.info("\nNext step: Train the CRNN model")
    logger.info("  python train_on_real_handwriting.py --epochs 100")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert IAM Handwriting Database to CRNN training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire dataset
  python convert_iam_dataset.py --input ./archive/iam_words --output ./data
  
  # Convert only first 1000 images (for quick test)
  python convert_iam_dataset.py --input ./archive/iam_words --output ./data --max-samples 1000
  
  # Change validation split (30% validation)
  python convert_iam_dataset.py --input ./archive/iam_words --output ./data --val-split 0.3
        """
    )
    
    parser.add_argument(
        "--input",
        default="./archive/iam_words",
        help="Input IAM archive directory (default: ./archive/iam_words)"
    )
    parser.add_argument(
        "--output",
        default="./data",
        help="Output directory for training data (default: ./data)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to convert (None = all, useful for testing)"
    )
    
    args = parser.parse_args()
    
    success = convert_iam_dataset(
        iam_archive_dir=args.input,
        output_dir=args.output,
        val_split=args.val_split,
        max_samples=args.max_samples
    )
    
    sys.exit(0 if success else 1)
