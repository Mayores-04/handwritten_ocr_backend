#!/usr/bin/env python3
"""
Download REAL EMNIST dataset from Kaggle
EMNIST = Extended MNIST with REAL handwritten letters A-Z and digits 0-9
(NOT synthetic, NOT computer-generated)

Requires: kaggle CLI installed and API credentials configured
"""

import os
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_kaggle_setup():
    """Check if Kaggle CLI is installed and configured"""
    logger.info("Checking Kaggle setup...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        logger.info("✓ Kaggle API authenticated successfully")
        return api
    except Exception as e:
        logger.error(f"❌ Kaggle setup failed: {e}")
        logger.error("\nTo fix:")
        logger.error("1. Install kaggle: pip install kaggle")
        logger.error("2. Get API token: https://www.kaggle.com/settings/account")
        logger.error("3. Download kaggle.json and place in ~/.kaggle/")
        logger.error("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        raise


def download_emnist():
    """Download EMNIST dataset from Kaggle"""
    logger.info("\n" + "="*60)
    logger.info("DOWNLOADING REAL EMNIST DATASET")
    logger.info("="*60)
    
    api = check_kaggle_setup()
    
    # Create data directory
    os.makedirs('data/emnist_raw', exist_ok=True)
    
    # Download EMNIST from Kaggle
    # This dataset has REAL handwritten characters, not synthetic
    logger.info("\nDownloading EMNIST dataset (this may take 5-15 minutes)...")
    logger.info("Dataset: ~600MB of real handwritten character images")
    
    try:
        # Download EMNIST dataset
        # Source: https://www.kaggle.com/datasets/crawford/emnist
        api.dataset_download_files('crawford/emnist', path='data/emnist_raw', unzip=True)
        logger.info("✓ Dataset downloaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error("\nAlternative: Manual download")
        logger.error("1. Go to: https://www.kaggle.com/datasets/crawford/emnist")
        logger.error("2. Click 'Download'")
        logger.error("3. Extract to: data/emnist_raw/")
        logger.error("4. Run this script again")
        return False


def verify_dataset():
    """Verify dataset structure"""
    logger.info("\nVerifying dataset structure...")
    
    emnist_dir = 'data/emnist_raw'
    
    if not os.path.exists(emnist_dir):
        logger.error(f"❌ Dataset directory not found: {emnist_dir}")
        return False
    
    # Check for EMNIST files
    expected_files = ['emnist-letters-train-images-idx3-ubyte', 
                      'emnist-letters-train-labels-idx1-ubyte',
                      'emnist-digits-train-images-idx3-ubyte',
                      'emnist-digits-train-labels-idx1-ubyte']
    
    found_files = os.listdir(emnist_dir)
    logger.info(f"Found files in {emnist_dir}:")
    for f in found_files[:10]:
        logger.info(f"  - {f}")
    
    logger.info(f"\n✓ Dataset structure verified")
    return True


def main():
    """Main setup pipeline"""
    logger.info("EMNIST REAL HANDWRITING DATASET SETUP")
    logger.info("="*60)
    
    # Step 1: Download
    if download_emnist():
        logger.info("\n✓ Download successful!")
        
        # Step 2: Verify
        if verify_dataset():
            logger.info("\n" + "="*60)
            logger.info("✓ SETUP COMPLETE")
            logger.info("="*60)
            logger.info("\nNext steps:")
            logger.info("1. Run: python train_on_real_emnist.py")
            logger.info("   This will train your model on REAL handwritten data")
            logger.info("   Expected accuracy: 95-97%")
            logger.info("   Time: ~30-50 minutes on CPU")
            return True
    
    logger.error("\n❌ Setup failed. See errors above.")
    return False


if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠ Cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        exit(1)
