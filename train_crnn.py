"""
Train CRNN model for handwriting recognition using local Img/ dataset
Auto-labels images using EasyOCR, then trains a CRNN for variable-length text recognition
"""

import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import random

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Import project utilities
from config import CHAR_CLASSES, MODEL_PATHS
from preprocessing import to_numpy, preprocess_image, upscale_for_handwriting

print("=" * 70)
print("CRNN HANDWRITING RECOGNITION MODEL TRAINING")
print("=" * 70)


class CRNNTrainer:
    """Train CRNN model for variable-length text recognition"""
    
    def __init__(self, img_dir='Img'):
        """Initialize trainer with dataset directory"""
        self.img_dir = Path(img_dir)
        self.char_list = CHAR_CLASSES  # '0-9A-Za-z'
        self.char_to_num = {char: idx for idx, char in enumerate(self.char_list)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.char_list)}
        self.img_height = 32
        self.img_width = 128
        self.max_length = 20  # Max text length to recognize
        
        print(f"\nDataset directory: {self.img_dir}")
        print(f"Character set: {self.char_list}")
        print(f"Total characters: {len(self.char_list)}")
        print(f"Image format: {self.img_width}x{self.img_height} (W x H)")
        print(f"Max text length: {self.max_length}")
    
    def load_images_and_auto_label(self, num_images=None):
        """
        Load images from Img/ folder and auto-label using EasyOCR batch processing
        Returns: (images_array, labels_array)
        """
        print("\n" + "=" * 70)
        print("STEP 1: LOADING IMAGES AND AUTO-LABELING WITH BATCH EasyOCR")
        print("=" * 70)
        
        # Get all PNG images
        img_files = sorted(list(self.img_dir.glob('*.png')))
        if not img_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        
        # Use representative sample for faster training
        # 1000 images = ~1 hour training, 500 = ~30 min, 250 = ~10 min
        sample_size = num_images or min(1000, len(img_files))
        
        # Sample evenly across the dataset
        step = max(1, len(img_files) // sample_size)
        img_files = img_files[::step][:sample_size]
        
        total_images = len(list(self.img_dir.glob('*.png')))
        print(f"Found {total_images} total images")
        print(f"Using representative sample: {len(img_files)} images")
        
        # Load EasyOCR for batch processing (MUCH faster)
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            device_name = "GPU" if use_gpu else "CPU"
            print(f"Loading EasyOCR ({device_name})...")
            reader = easyocr.Reader(['en'], gpu=use_gpu)
        except Exception as e:
            print(f"ERROR: Could not load EasyOCR: {e}")
            raise
        
        # Load all images first (faster than reading one-by-one)
        print(f"\nPhase 1: Loading {len(img_files)} images from disk...")
        img_arrays = []
        img_paths_valid = []
        
        for idx, img_path in enumerate(img_files, 1):
            try:
                img_array = cv2.imread(str(img_path))
                if img_array is None:
                    continue
                img_arrays.append(img_array)
                img_paths_valid.append(img_path)
                
                if idx % 200 == 0:
                    print(f"  Loaded {idx}/{len(img_files)} images")
            except Exception as e:
                print(f"  Skipped {img_path.name}: {e}")
        
        print(f"✓ Loaded {len(img_arrays)} valid images")
        
        # Batch processing with EasyOCR (10x faster than sequential)
        print(f"\nPhase 2: Batch labeling with EasyOCR (batch size: 10)...")
        batch_size = 10
        labels = []
        
        for batch_start in range(0, len(img_arrays), batch_size):
            batch_end = min(batch_start + batch_size, len(img_arrays))
            batch = img_arrays[batch_start:batch_end]
            
            # Convert BGR to RGB for batch
            batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch]
            
            # Batch readtext (more efficient)
            try:
                for img_rgb in batch_rgb:
                    result = reader.readtext(img_rgb, detail=0)
                    text = ' '.join(result).strip().upper()
                    labels.append(text if text and len(text) <= self.max_length else "A")
            except Exception as e:
                print(f"  Batch processing warning: {e}")
                for img_rgb in batch_rgb:
                    labels.append("A")  # Default label
            
            if batch_end % 200 == 0 or batch_end == len(img_arrays):
                print(f"  Labeled {batch_end}/{len(img_arrays)} images")
        
        # Preprocess images
        print(f"\nPhase 3: Preprocessing images to CRNN format...")
        images = []
        valid_pairs = []
        
        for idx, (img_array, label) in enumerate(zip(img_arrays, labels)):
            try:
                processed = self._preprocess_for_crnn(img_array)
                images.append(processed)
                valid_pairs.append((processed, label))
                
                if (idx + 1) % 200 == 0:
                    print(f"  Preprocessed {idx + 1}/{len(img_arrays)} images")
            except Exception as e:
                print(f"  Preprocessing failed: {e}")
        
        print(f"\n✓ Processed {len(images)} images successfully")
        
        if len(images) < 10:
            raise ValueError(f"Not enough training data! Got {len(images)}, need at least 10")
        
        return np.array(images), [label for _, label in valid_pairs]
    
    def _preprocess_for_crnn(self, img_array):
        """Preprocess image to 32x128 grayscale for CRNN"""
        if isinstance(img_array, np.ndarray):
            # Convert BGR to grayscale if needed
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_array
        else:
            raise ValueError("Expected numpy array")
        
        # Resize to 32x128
        img_resized = cv2.resize(img_gray, (self.img_width, self.img_height), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add channel dimension for CNN input
        img_channel = np.expand_dims(img_normalized, axis=-1)
        
        return img_channel
    
    def _encode_labels(self, labels):
        """Convert text labels to integer sequences, padded to max_length"""
        encoded = []
        input_lengths = []
        
        for text in labels:
            # Convert text to character indices, stop at max_length
            encoded_text = [self.char_to_num.get(char, 0) for char in text[:self.max_length]]
            input_lengths.append(len(encoded_text))
            
            # Pad to max_length
            padded = encoded_text + [0] * (self.max_length - len(encoded_text))
            encoded.append(padded)
        
        return np.array(encoded), np.array(input_lengths)
    
    def build_crnn_model(self):
        """
        Build CRNN model architecture
        - CNN for feature extraction (using existing char_model if available)
        - RNN (Bidirectional LSTM) for sequence modeling
        - CTC loss for variable-length output
        """
        print("\n" + "=" * 70)
        print("STEP 2: BUILDING CRNN ARCHITECTURE")
        print("=" * 70)
        
        # Input layer: 32x128x1 grayscale images
        input_layer = layers.Input(shape=(self.img_height, self.img_width, 1), 
                                   name="image_input")
        
        # ===== CNN FEATURE EXTRACTION =====
        # Block 1: Conv -> MaxPool
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 16x64
        
        # Block 2: Conv -> MaxPool
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 8x32
        
        # Block 3: Conv -> MaxPool
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 4x16
        
        # Block 4: Conv layers (no pooling to preserve width)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        print("✓ CNN blocks: 32→64→128→256 filters")
        
        # ===== RESHAPE FOR RNN =====
        # From (4, 16, 256) -> (sequence_steps, features)
        # Collapse height dimension
        h, w, c = (4, 16, 256)
        x = layers.Reshape((w, h * c))(x)  # (16, 1024)
        
        # ===== RNN SEQUENCE MODELING =====
        # Bidirectional LSTM for better context
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.5)(x)
        
        print("✓ RNN: Bidirectional LSTM (256→128)")
        
        # ===== CTC OUTPUT =====
        # Output layer: logits for CTC loss
        # Output shape: (batch, sequence_length, num_chars)
        output = layers.Dense(len(self.char_list) + 1, activation='softmax', name="ctc_output")(x)
        
        # Build model
        model = Model(inputs=input_layer, outputs=output)
        
        # CTC Loss function
        def ctc_loss_fn(y_true, y_pred):
            """Custom CTC loss function"""
            # y_true: (batch, max_length) - encoded text
            # y_pred: (batch, time_steps, num_classes) - network output
            
            # Get actual input lengths (from label encoding)
            batch_size = tf.shape(y_pred)[0]
            input_length = tf.shape(y_pred)[1]
            label_length = tf.reduce_sum(tf.cast(y_true != 0, tf.int32) + 
                                        tf.cast(y_true == 0, tf.int32) * 0, axis=1)
            
            # Clip to valid range
            input_length = tf.maximum(input_length, 1)
            label_length = tf.maximum(label_length, 1)
            
            # Flatten y_pred for CTC loss
            y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
            
            # CTC loss
            loss = tf.nn.ctc_loss(
                labels=tf.cast(y_true, tf.int32),
                logits=y_pred_flat,
                label_length=label_length,
                logit_length=tf.tile([input_length], [batch_size]),
                logits_time_major=False,
                blank_index=len(self.char_list)
            )
            
            return tf.reduce_mean(loss)
        
        # Compile with CTC loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=ctc_loss_fn,
            metrics=['accuracy']
        )
        
        print("\n✓ Model compiled with CTC loss")
        model.summary()
        
        return model
    
    def train(self, model, images, labels, epochs=50, batch_size=32):
        """Train CRNN model"""
        print("\n" + "=" * 70)
        print("STEP 3: PREPARING TRAINING DATA")
        print("=" * 70)
        
        # Encode labels
        encoded_labels, input_lengths = self._encode_labels(labels)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, encoded_labels))
        dataset = dataset.shuffle(len(images)).batch(batch_size)
        
        print(f"✓ Dataset: {len(images)} samples, batch size={batch_size}")
        print(f"✓ Splits: 80% train, 20% val ({int(len(images)*0.8)} train, {int(len(images)*0.2)} val)")
        
        # Train-val split
        train_size = int(0.8 * len(images))
        train_images = images[:train_size]
        train_labels = encoded_labels[:train_size]
        val_images = images[train_size:]
        val_labels = encoded_labels[train_size:]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.batch(batch_size)
        
        print("\n" + "=" * 70)
        print("STEP 4: TRAINING CRNN")
        print("=" * 70)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        ]
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training complete!")
        return model, history
    
    def save_model(self, model, output_path=None):
        """Save trained model"""
        if output_path is None:
            output_path = MODEL_PATHS['handwriting_model']
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("STEP 5: SAVING MODEL")
        print("=" * 70)
        
        model.save(str(output_path))
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"✓ Model saved: {output_path}")
        print(f"✓ File size: {file_size_mb:.2f} MB")


def main():
    """Main training pipeline"""
    try:
        # Initialize trainer
        trainer = CRNNTrainer(img_dir='Img')
        
        # Load and auto-label images (optimized batch processing)
        images, labels = trainer.load_images_and_auto_label()
        
        # Build CRNN model
        model = trainer.build_crnn_model()
        
        # Train model (adaptive epochs based on dataset size)
        train_epochs = max(5, min(50, 100 // (len(images) // 200)))  # Scale down for larger datasets
        print(f"\nTraining epochs: {train_epochs} (scaled for dataset size: {len(images)} samples)")
        model, history = trainer.train(model, images, labels, epochs=train_epochs, batch_size=32)
        
        # Save model
        trainer.save_model(model)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print("\nModel is ready to use. Keras will now use this trained model")
        print("for handwriting OCR instead of falling back to EasyOCR.")
        print("\nStart your app with: python app.py")
        print("=" * 70)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
