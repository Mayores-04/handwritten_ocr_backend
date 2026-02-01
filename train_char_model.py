"""
Training script for handwritten character recognition using the english.csv dataset
Uses Keras CNN for character-level recognition
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Character mappings
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Characters: {CHARS}")


def load_dataset(csv_path, img_base_path=None, img_size=(32, 32)):
    """
    Load dataset from CSV file
    
    Args:
        csv_path: Path to the CSV file with columns 'image' and 'label'
        img_base_path: Base path for images (if different from CSV location)
        img_size: Target image size (height, width)
    
    Returns:
        images: numpy array of images
        labels: numpy array of labels
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from CSV")
    
    if img_base_path is None:
        img_base_path = os.path.dirname(csv_path)
    
    images = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        img_path = os.path.join(img_base_path, row['image'])
        label = str(row['label'])
        
        # Skip if label not in our character set
        if label not in CHAR_TO_IDX:
            skipped += 1
            continue
        
        # Load and preprocess image
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to target size
                img = cv2.resize(img, img_size)
                # Normalize to 0-1
                img = img.astype('float32') / 255.0
                # Invert if needed (white text on black background)
                if np.mean(img) > 0.5:
                    img = 1.0 - img
                
                images.append(img)
                labels.append(CHAR_TO_IDX[label])
        else:
            skipped += 1
    
    print(f"Loaded {len(images)} images, skipped {skipped}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    
    return images, labels


def create_character_cnn(input_shape=(32, 32, 1), num_classes=62):
    """
    Create a CNN model for character recognition
    
    Architecture optimized for handwritten character recognition
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_augmentation():
    """
    Create data augmentation pipeline for handwritten characters
    """
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])


def train_model(csv_path, epochs=50, batch_size=32, save_path='models/char_model.keras'):
    """
    Train the character recognition model
    """
    print("Loading dataset...")
    images, labels = load_dataset(csv_path)
    
    print(f"Dataset shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    print("\nCreating model...")
    model = create_character_cnn(input_shape=images.shape[1:], num_classes=NUM_CLASSES)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Plot training history
    plot_history(history, save_path.replace('.keras', '_history.png'))
    
    return model, history


def plot_history(history, save_path=None):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def predict_character(model, image):
    """
    Predict a single character from image
    
    Args:
        model: Trained Keras model
        image: PIL Image or numpy array
    
    Returns:
        predicted_char: Predicted character
        confidence: Prediction confidence
    """
    # Preprocess
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    
    # Resize
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    
    # Invert if needed
    if np.mean(image) > 0.5:
        image = 1.0 - image
    
    # Add dimensions
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    
    # Predict
    predictions = model.predict(image, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    return IDX_TO_CHAR[predicted_idx], confidence


def test_model(model_path, csv_path, num_samples=10):
    """
    Test the model on some samples
    """
    model = keras.models.load_model(model_path)
    images, labels = load_dataset(csv_path)
    
    # Random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    correct = 0
    for idx in indices:
        img = images[idx]
        true_label = IDX_TO_CHAR[labels[idx]]
        
        # Predict
        pred_label, conf = predict_character(model, img[:,:,0] * 255)
        
        is_correct = pred_label == true_label
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"{status} True: {true_label}, Predicted: {pred_label}, Confidence: {conf:.2f}")
    
    print(f"\nAccuracy on {num_samples} samples: {correct/num_samples:.2%}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train handwritten character recognition model')
    parser.add_argument('--csv', type=str, default='english.csv', help='Path to CSV dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='models/char_model.keras', help='Output model path')
    parser.add_argument('--test', action='store_true', help='Test mode - evaluate existing model')
    
    args = parser.parse_args()
    
    if args.test:
        test_model(args.output, args.csv)
    else:
        train_model(args.csv, args.epochs, args.batch_size, args.output)
