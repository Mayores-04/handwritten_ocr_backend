#!/usr/bin/env python3
"""
Train OCR model using REAL EMNIST dataset
(Handwritten characters A-Z and digits 0-9 from real people)

This replaces the synthetic data approach with actual handwritten samples.
Expected accuracy: 95-97% on real handwritten characters
"""

import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import CHAR_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_CLASSES = len(CHAR_CLASSES)


def load_emnist_data(emnist_dir='data/emnist_raw'):
    """Load EMNIST data from CSV files"""
    logger.info("Loading REAL EMNIST dataset from CSV...")
    
    import pandas as pd
    
    # EMNIST byclass has all characters (digits 0-9 and letters A-Z)
    csv_file = os.path.join(emnist_dir, 'emnist-byclass-train.csv')
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file not found: {csv_file}")
    
    logger.info(f"  Loading {csv_file}...")
    
    # CSV format: label, pixel1, pixel2, ..., pixel784 (for 28x28 images)
    df = pd.read_csv(csv_file, header=None)
    
    # Extract labels and images
    y_raw = df.iloc[:, 0].values  # First column is label
    X_train = df.iloc[:, 1:].values  # Rest are pixel values
    
    logger.info(f"  Loaded {len(X_train)} samples")
    logger.info(f"  Label range: {y_raw.min()}-{y_raw.max()}")
    
    # EMNIST byclass labels: 0-9 (digits), 10-35 (letters A-Z)
    # Filter to only valid classes
    valid_mask = (y_raw >= 0) & (y_raw < NUM_CLASSES)
    X_train = X_train[valid_mask]
    y_train = y_raw[valid_mask]
    
    logger.info(f"  After filtering: {len(X_train)} valid samples")
    
    # Reshape images from flat (784,) to (28, 28)
    X_train = X_train.reshape(-1, 28, 28)
    
    logger.info(f"Combined: {len(X_train)} total samples")
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    
    # Reshape to include channel
    X_train = X_train.reshape(-1, 28, 28, 1)
    
    # One-hot encode
    y_train_encoded = keras.utils.to_categorical(y_train.astype(int), NUM_CLASSES)
    
    # Split into train/val
    split_idx = int(len(X_train) * 0.9)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train_encoded[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train_encoded[split_idx:]
    
    logger.info(f"Train: {len(X_train_split)}, Val: {len(X_val)}")
    logger.info(f"Classes: {NUM_CLASSES} (0-9 digits, 10-35 letters A-Z)")
    
    return X_train_split, y_train_split, X_val, y_val


def build_model():
    """Build CNN for handwritten character recognition"""
    logger.info("Building model...")
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model built with {model.count_params()} parameters")
    return model


def train_model(X_train, y_train, X_val, y_val):
    """Train on real EMNIST data"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING ON REAL EMNIST DATA")
    logger.info("="*60)
    
    model = build_model()
    
    # Callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(
        'models/char_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train
    logger.info("\nTraining (this will take 30-50 minutes)...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=40,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"\n" + "="*60)
    logger.info(f"✓ Training complete!")
    logger.info(f"  Validation Loss: {val_loss:.4f}")
    logger.info(f"  Validation Accuracy: {val_acc:.2%}")
    logger.info(f"="*60)
    
    return model, history


def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("REAL EMNIST HANDWRITING MODEL TRAINING")
    logger.info("="*60)
    
    os.makedirs('models', exist_ok=True)
    
    # Load real data
    logger.info("\n[1/2] Loading REAL EMNIST dataset...")
    try:
        X_train, y_train, X_val, y_val = load_emnist_data()
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        logger.error("\nTo fix:")
        logger.error("1. Run: python setup_emnist_real.py")
        logger.error("2. Then run this script again")
        return False
    
    # Train
    logger.info("\n[2/2] Training model...")
    try:
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        logger.info("\n" + "="*60)
        logger.info("✓ TRAINING PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info("\nYour improved model is ready!")
        logger.info("Run: python app.py")
        logger.info("\nExpected results:")
        logger.info("  ✓ Recognition of digits 0-9")
        logger.info("  ✓ Recognition of letters A-Z")
        logger.info("  ✓ Accuracy: 95-97% on handwritten characters")
        
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False


if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠ Training cancelled by user")
        exit(1)
