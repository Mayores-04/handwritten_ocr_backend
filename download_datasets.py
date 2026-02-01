"""
Additional Handwritten Datasets for OCR Training

This script provides utilities to download and prepare additional
handwritten text datasets for training the OCR model.

Available Datasets:
1. EMNIST (Extended MNIST) - Letters and digits
2. IAM Handwriting Database - Full words and sentences
3. MNIST - Handwritten digits
4. Chars74K - Characters from natural images
"""

import os
import numpy as np
import requests
import zipfile
import gzip
from tqdm import tqdm


def download_file(url, save_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_emnist(save_dir='data/emnist'):
    """
    Download EMNIST ByClass dataset
    Contains 62 classes: 0-9, A-Z, a-z
    814,255 total samples
    
    More info: https://www.nist.gov/itl/products-and-services/emnist-dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Downloading EMNIST ByClass dataset...")
    print("This dataset contains handwritten characters: 0-9, A-Z, a-z")
    
    # EMNIST is available through TensorFlow datasets
    try:
        import tensorflow_datasets as tfds
        
        print("Loading EMNIST via TensorFlow Datasets...")
        ds, info = tfds.load('emnist/byclass', split='train', with_info=True)
        
        print(f"Dataset info: {info}")
        print(f"Number of classes: {info.features['label'].num_classes}")
        
        return ds, info
        
    except ImportError:
        print("tensorflow_datasets not installed. Installing...")
        os.system('pip install tensorflow-datasets')
        return download_emnist(save_dir)


def download_mnist(save_dir='data/mnist'):
    """
    Download MNIST dataset (handwritten digits 0-9)
    60,000 training samples, 10,000 test samples
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Downloading MNIST dataset...")
    
    try:
        from tensorflow import keras
        
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Image shape: {x_train[0].shape}")
        
        # Save to numpy files
        np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        
        print(f"Saved to {save_dir}")
        
        return (x_train, y_train), (x_test, y_test)
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def prepare_iam_dataset(data_path='data/iam'):
    """
    Prepare IAM Handwriting Database
    
    Note: IAM requires registration to download.
    Register at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
    
    After downloading, extract to data/iam/
    """
    print("IAM Handwriting Database")
    print("=" * 50)
    print("This dataset contains full handwritten text samples.")
    print("")
    print("To use IAM dataset:")
    print("1. Register at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
    print("2. Download the 'words' dataset")
    print("3. Extract to: data/iam/")
    print("")
    print("Expected structure:")
    print("  data/iam/")
    print("  ├── words/")
    print("  │   ├── a01/")
    print("  │   │   ├── a01-000u/")
    print("  │   │   │   ├── a01-000u-00-00.png")
    print("  │   │   │   └── ...")
    print("  │   └── ...")
    print("  └── words.txt")
    
    if os.path.exists(os.path.join(data_path, 'words.txt')):
        print("\n✓ IAM dataset found!")
        return True
    else:
        print("\n✗ IAM dataset not found. Please download manually.")
        return False


def create_synthetic_handwriting_data(num_samples=5000, save_dir='data/synthetic'):
    """
    Create synthetic handwriting-like data for training
    Uses different fonts and augmentation to simulate handwriting
    """
    os.makedirs(save_dir, exist_ok=True)
    
    from PIL import Image, ImageDraw, ImageFont
    import random
    import cv2
    
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    
    images = []
    labels = []
    
    print(f"Creating {num_samples} synthetic handwriting samples...")
    
    for i in tqdm(range(num_samples)):
        # Random character
        char = random.choice(chars)
        
        # Create image
        img_size = 64
        img = Image.new('L', (img_size, img_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Random font size
        font_size = random.randint(36, 52)
        
        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Random position with slight offset
        x = random.randint(5, 20)
        y = random.randint(5, 15)
        
        # Draw character
        draw.text((x, y), char, fill=0, font=font)
        
        # Convert to numpy
        img_array = np.array(img)
        
        # Add noise
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Random rotation
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((img_size/2, img_size/2), angle, 1)
        img_array = cv2.warpAffine(img_array, M, (img_size, img_size), borderValue=255)
        
        # Random blur
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        # Resize to 32x32
        img_array = cv2.resize(img_array, (32, 32))
        
        images.append(img_array)
        labels.append(char)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Save
    np.save(os.path.join(save_dir, 'images.npy'), images)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    
    # Also save as CSV
    import pandas as pd
    
    img_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    data = []
    for i, (img, label) in enumerate(zip(images, labels)):
        img_path = f'images/img_{i:05d}.png'
        cv2.imwrite(os.path.join(save_dir, img_path), img)
        data.append({'image': img_path, 'label': label})
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'synthetic.csv'), index=False)
    
    print(f"\nSaved {len(images)} samples to {save_dir}")
    print(f"  - images.npy: {images.shape}")
    print(f"  - labels.npy: {labels.shape}")
    print(f"  - synthetic.csv: CSV format")
    
    return images, labels


def combine_datasets(datasets, save_path='data/combined.csv'):
    """
    Combine multiple datasets into one CSV file
    """
    import pandas as pd
    
    all_data = []
    
    for csv_path in datasets:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add base path
            base_path = os.path.dirname(csv_path)
            df['image'] = df['image'].apply(lambda x: os.path.join(base_path, x))
            all_data.append(df)
            print(f"Added {len(df)} samples from {csv_path}")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(save_path, index=False)
    
    print(f"\nCombined dataset: {len(combined)} samples")
    print(f"Saved to {save_path}")
    
    return combined


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download additional handwriting datasets')
    parser.add_argument('--dataset', type=str, choices=['emnist', 'mnist', 'iam', 'synthetic', 'all'],
                        default='synthetic', help='Dataset to download')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of synthetic samples')
    parser.add_argument('--save-dir', type=str, default='data', help='Save directory')
    
    args = parser.parse_args()
    
    if args.dataset == 'mnist' or args.dataset == 'all':
        download_mnist(os.path.join(args.save_dir, 'mnist'))
    
    if args.dataset == 'emnist' or args.dataset == 'all':
        download_emnist(os.path.join(args.save_dir, 'emnist'))
    
    if args.dataset == 'iam' or args.dataset == 'all':
        prepare_iam_dataset(os.path.join(args.save_dir, 'iam'))
    
    if args.dataset == 'synthetic' or args.dataset == 'all':
        create_synthetic_handwriting_data(args.num_samples, os.path.join(args.save_dir, 'synthetic'))
