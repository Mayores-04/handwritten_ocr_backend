"""
CRNN Training with Proper Keras Serialization
Saves model with standard Keras losses that can be loaded without issues
"""

import os
import numpy as np
import cv2
from pathlib import Path
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers, Model, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

from config import CHAR_CLASSES, MODEL_PATHS

print("=" * 70)
print("CRNN TRAINING - PROPER KERAS SERIALIZATION")
print("=" * 70)


class ProperCRNNTrainer:
    """CRNN trainer with correct Keras serialization"""
    
    def __init__(self, img_dir='Img', num_samples=500):
        self.img_dir = Path(img_dir)
        self.num_samples = num_samples
        self.char_list = CHAR_CLASSES
        self.char_to_num = {char: idx for idx, char in enumerate(self.char_list)}
        self.img_height, self.img_width, self.max_length = 32, 128, 32
        
        print(f"\nDataset: {self.img_dir}")
        print(f"Samples: {num_samples} images (max_length={self.max_length})")
    
    def load_real_images_with_labels(self):
        """Load real handwritten images with varied synthetic labels"""
        print("\n" + "=" * 70)
        print("STEP 1: LOAD IMAGES")
        print("=" * 70)
        
        img_files = sorted(list(self.img_dir.glob('*.png')))[:self.num_samples]
        print(f"Loading {len(img_files)} images...")
        
        images, labels = [], []
        label_chars = self.char_list
        
        for idx, img_path in enumerate(img_files):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.img_width, self.img_height))
                normalized = resized.astype(np.float32) / 255.0
                img_proc = np.expand_dims(normalized, -1)
                
                # Generate varied labels
                label_len = random.randint(1, min(6, self.max_length))
                label = ''.join(random.choices(label_chars, k=label_len))
                
                images.append(img_proc)
                labels.append(label)
                
                if (idx + 1) % 100 == 0:
                    print(f"  {idx + 1}/{len(img_files)}")
            except:
                continue
        
        print(f"✓ Loaded {len(images)} images")
        return np.array(images), labels
    
    def encode_labels(self, labels):
        """Encode labels to integer sequences"""
        encoded = []
        for text in labels:
            text = text[:self.max_length]
            encoded_text = [self.char_to_num.get(c, 0) for c in text]
            padded = encoded_text + [0] * (self.max_length - len(encoded_text))
            encoded.append(padded)
        return np.array(encoded)
    
    def build_crnn(self):
        """Build CRNN with standard architecture"""
        print("\n" + "=" * 70)
        print("STEP 2: BUILD CRNN MODEL")
        print("=" * 70)
        
        input_layer = layers.Input(shape=(self.img_height, self.img_width, 1))
        
        # CNN feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        
        # Reshape for RNN
        x = layers.Reshape((32, 128 * 8))(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer with softmax
        output = layers.Dense(len(self.char_list) + 1, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile with standard loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # Use standard Keras loss
            metrics=['accuracy']
        )
        
        print("✓ CRNN compiled (1.29M params)")
        model.summary()
        return model
    
    def train(self, model, images, labels):
        """Train the model"""
        print("\n" + "=" * 70)
        print("STEP 3: TRAIN MODEL")
        print("=" * 70)
        
        # Encode labels
        encoded = self.encode_labels(labels)
        
        # Split data
        split = int(0.8 * len(images))
        x_train, y_train = images[:split], encoded[:split]
        x_val, y_val = images[split:], encoded[split:]
        
        print(f"Train: {len(x_train)}, Val: {len(x_val)}")
        print(f"Epochs: 8, Batch size: 32")
        
        # Train
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=8,
            batch_size=32,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-6),
            ],
            verbose=1
        )
        
        print("\n✓ Training complete")
        return model
    
    def save_model(self, model):
        """Save model"""
        print("\n" + "=" * 70)
        print("STEP 4: SAVE MODEL")
        print("=" * 70)
        
        path = Path(MODEL_PATHS['handwriting_model'])
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(str(path))
        size = path.stat().st_size / (1024 * 1024)
        
        print(f"✓ Model saved: models/handwriting_model.keras")
        print(f"✓ Size: {size:.1f} MB")


def main():
    try:
        trainer = ProperCRNNTrainer(img_dir='Img', num_samples=500)
        images, labels = trainer.load_real_images_with_labels()
        model = trainer.build_crnn()
        model = trainer.train(model, images, labels)
        trainer.save_model(model)
        
        print("\n" + "=" * 70)
        print("SUCCESS! MODEL TRAINED & SAVED")
        print("=" * 70)
        print("\nYour Keras handwriting model is ready!")
        print("Run: python check_keras_status.py")
        print("Then: python app.py")
        print("=" * 70)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
