"""
Train CRNN model with proper CTC loss for full-document handwriting OCR

This trains per-LINE recognition (not full pages). Each line image is segmented 
from the document and trained with real CTC loss for variable-length sequences.

Supports:
- IAM Handwriting Database (lines) - recommended: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
- HWDB - Chinese Handwriting Database
- Custom labeled line images

Dataset structure:
    data/
      train/
        line_001.png
        line_002.png
        ...
      val/
        line_100.png
        ...
      labels.txt    (format: filename,transcription)

Installation:
    pip install tensorflow keras pillow opencv-python numpy editdistance

Usage:
    # See dataset preparation guide
    python train_on_real_handwriting.py --guide

    # Train on your dataset
    python train_on_real_handwriting.py --dataset-path ./data --epochs 100 --batch-size 32
"""

import os
import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2
try:
    from keras.saving import register_keras_serializable
except ImportError:
    from keras.utils import register_keras_serializable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
except ImportError:
    logger.error("TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


# ==================== Character Encoding ====================

class CharacterSet:
    """Encode/decode characters for handwriting recognition"""
    
    def __init__(self):
        # Common character set for English handwriting + digits + punctuation
        self.chars = (
            ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789.,;:!?\'"()-'
        )
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        # Reserve one extra class for CTC blank at the last index.
        self.blank_index = len(self.chars)
        
    def encode_text(self, text: str) -> np.ndarray:
        """Convert text string to character indices"""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Unknown character → space
                encoded.append(self.char_to_idx[' '])
        return np.array(encoded, dtype=np.int32)
    
    def decode_predictions(self, pred: np.ndarray) -> str:
        """Decode model output (argmax) back to text with CTC post-processing"""
        # Remove consecutive duplicates
        decoded = []
        prev_idx = -1
        for idx in np.argmax(pred, axis=1):
            if idx != prev_idx and idx != self.blank_index:
                decoded.append(self.idx_to_char.get(int(idx), ''))
            prev_idx = idx
        return ''.join(decoded)
    
    @property
    def num_classes(self) -> int:
        """Number of classes, including one extra CTC blank class."""
        return len(self.chars) + 1


# ==================== Dataset Loading ====================

class HandwritingLineDataset:
    """Load handwritten line images with text transcriptions"""
    
    def __init__(self, dataset_path: str, test_size: float = 0.2):
        self.dataset_path = Path(dataset_path)
        self.test_size = test_size
        self.charset = CharacterSet()
        
        self.train_images = []
        self.train_texts = []
        self.val_images = []
        self.val_texts = []
        
    def load_dataset(self) -> bool:
        """Load images and labels from directory"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        labels_file = self.dataset_path / "labels.txt"
        if not labels_file.exists():
            logger.error(f"labels.txt not found at {labels_file}")
            logger.error("Expected format: filename.png,transcription text")
            return False
        
        # Read labels
        label_dict = {}
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        text = parts[1].strip()
                        label_dict[filename] = text
        
        logger.info(f"Found {len(label_dict)} labeled samples")
        
        # Load images from train and val directories
        all_samples = []
        unreadable_files = []
        progress_log_every = 2000
        
        for data_dir in [self.dataset_path / "train", self.dataset_path / "val"]:
            if not data_dir.exists():
                logger.warning(f"Directory not found: {data_dir}")
                continue

            image_paths = sorted(data_dir.glob("*.png")) + sorted(data_dir.glob("*.jpg"))
            logger.info(f"Scanning {len(image_paths)} images in {data_dir}")

            for img_idx, img_path in enumerate(image_paths, start=1):
                filename = img_path.name
                base_name = img_path.stem
                
                # Try to find label
                text = label_dict.get(filename) or label_dict.get(base_name)
                if not text:
                    logger.warning(f"No label for {filename}")
                    continue
                
                # Load image
                try:
                    # Skip placeholders/corrupted files quickly before decode.
                    if img_path.stat().st_size == 0:
                        unreadable_files.append(str(img_path))
                        logger.warning(f"Skipped zero-byte image: {img_path}")
                        continue

                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        unreadable_files.append(str(img_path))
                        logger.warning(f"Skipped unreadable image: {img_path}")
                        continue
                    
                    all_samples.append((img, text))
                except Exception as e:
                    unreadable_files.append(str(img_path))
                    logger.warning(f"Skipped image with error {img_path}: {e}")

                if img_idx % progress_log_every == 0:
                    logger.info(
                        f"Load progress [{data_dir.name}] {img_idx}/{len(image_paths)} files | "
                        f"usable: {len(all_samples)} | skipped: {len(unreadable_files)}"
                    )
        
        if not all_samples:
            logger.error("No images loaded from dataset!")
            return False
        
        logger.info(f"Loaded {len(all_samples)} image-text pairs")
        if unreadable_files:
            logger.warning(f"Skipped {len(unreadable_files)} unreadable images during load")
            preview = unreadable_files[:5]
            for bad_path in preview:
                logger.warning(f"  - {bad_path}")
            if len(unreadable_files) > len(preview):
                logger.warning(f"  ... and {len(unreadable_files) - len(preview)} more")
        
        # Split into train/val
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * (1 - self.test_size))
        
        self.train_images = [x[0] for x in all_samples[:split_idx]]
        self.train_texts = [x[1] for x in all_samples[:split_idx]]
        self.val_images = [x[0] for x in all_samples[split_idx:]]
        self.val_texts = [x[1] for x in all_samples[split_idx:]]
        
        logger.info(f"Train: {len(self.train_images)}, Val: {len(self.val_images)}")
        return True
    
    def preprocess_image(self, img: np.ndarray, target_height: int = 32) -> np.ndarray:
        """Resize image maintaining aspect ratio, pad to fixed height"""
        h, w = img.shape
        
        # Calculate new width maintaining aspect ratio
        aspect = w / h
        new_h = target_height
        new_w = int(target_height * aspect)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def get_batch(self, indices: List[int], use_train: bool = True) -> Tuple:
        """Get a batch of images and corresponding text"""
        images = self.train_images if use_train else self.val_images
        texts = self.train_texts if use_train else self.val_texts
        
        batch_images = []
        batch_texts = []
        batch_text_lengths = []
        batch_input_lengths = []
        
        max_width = 512  # Maximum image width
        target_height = 32
        cnn_downsample_factor = 16  # width: /2 /2 /2 /2 across pooling blocks
        
        for idx in indices:
            # Preprocess image
            img = self.preprocess_image(images[idx], target_height)
            
            # Truncate/pad to max width
            h, w = img.shape  
            effective_w = min(w, max_width)
            if w > max_width:
                img = img[:, :max_width]
            else:
                # Pad right with white (1.0)
                padded = np.ones((h, max_width), dtype=np.float32)
                padded[:, :w] = img
                img = padded
            
            batch_images.append(img)
            
            # Encode text
            text = texts[idx]
            encoded = self.charset.encode_text(text)
            batch_texts.append(encoded)
            batch_text_lengths.append(len(encoded))
            
            # Input length after CNN pooling along width.
            input_len = max(1, effective_w // cnn_downsample_factor)
            batch_input_lengths.append(input_len)
        
        # Convert to numpy arrays with proper shapes
        batch_images = np.array(batch_images)[..., np.newaxis]  # Add channel dim
         
        # Pad text sequences to same length
        max_text_len = max(batch_text_lengths)
        padded_texts = np.zeros((len(batch_texts), max_text_len), dtype=np.int32)
        for i, text in enumerate(batch_texts):
            padded_texts[i, :len(text)] = text
        
        batch_text_lengths = np.array(batch_text_lengths, dtype=np.int32)
        batch_input_lengths = np.array(batch_input_lengths, dtype=np.int32)
        
        return (batch_images, padded_texts), (batch_text_lengths, batch_input_lengths)


# ==================== CRNN Model ====================

def build_crnn_model(charset: CharacterSet, img_height: int = 32) -> keras.Model:
    """
    Build CRNN (Convolutional Recurrent Neural Network) for handwriting recognition.
    
    Architecture:
    - CNN: Feature extraction (4 conv blocks with 2x2 max pooling)
    - RNN: Bidirectional LSTM for sequence modeling
    - Output: Character probabilities for each time step
    """
    input_img = layers.Input(shape=(img_height, None, 1), name='image')
    
    # ===== CNN Feature Extraction =====
    # Input: (batch, 32, variable_width, 1)
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)  # Height: 16
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Height: 8
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    # Block 4: 256 filters
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)  # Width is halved again
    
    # Block 5: 512 filters
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)  # Width is halved again
    
    # After pooling: (batch, 8, width/16, 512)
    # Reshape for RNN: (batch, width/16, 8*512=4096)
    x = layers.Reshape(target_shape=((-1, 512 * 8)))(x)
    
    # ===== RNN Layer =====
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer: predict character class for each time step
    # Output shape: (batch, width/16, num_classes)
    output = layers.Dense(charset.num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=input_img, outputs=output, name='CRNN')
    return model


# ==================== CTC Loss & Training ====================

@register_keras_serializable()
def ctc_loss_fn(y_true, y_pred):
    """
    TensorFlow CTC Loss function for variable-length sequences.
    
    Args:
        y_true: Tuple of (text_indices, text_lengths, input_lengths)
        y_pred: Model predictions (batch, time_steps, num_classes)
    """
    text_indices, text_lengths, input_lengths = y_true
    
    # Reshape to ensure proper 1D arrays
    text_lengths = tf.reshape(text_lengths, [-1])
    input_lengths = tf.reshape(input_lengths, [-1])
    
    # Use TensorFlow's native CTC loss
    # logits_time_major=False means (batch, time, classes)
    loss = tf.nn.ctc_loss(
        text_indices,
        y_pred,
        label_length=text_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=-1
    )
    
    return loss


class CTCCallback(keras.callbacks.Callback):
    """Custom callback for CTC-based training"""
    
    def __init__(self, dataset: HandwritingLineDataset, charset: CharacterSet):
        super().__init__()
        self.dataset = dataset
        self.charset = charset
    
    def on_epoch_end(self, epoch, logs=None):
        """Sample validation predictions every 10 epochs"""
        if epoch % 10 == 0 and epoch > 0:
            logger.info(f"\n[Epoch {epoch}] Sample predictions:")
            
            # Sample a few validation images
            val_indices = np.random.choice(len(self.dataset.val_images), 
                                          min(3, len(self.dataset.val_images)), 
                                          replace=False)
            (val_images, val_texts), (val_text_lens, val_input_lens) = \
                self.dataset.get_batch(val_indices.tolist(), use_train=False)
            
            # Predict
            predictions = self.model.predict(val_images, verbose=0)
            
            # Decode and print samples
            for i in range(len(val_images)):
                pred_text = self.charset.decode_predictions(predictions[i])
                true_text = self.dataset.val_texts[val_indices[i]]
                logger.info(f"  True: {true_text[:50]}")
                logger.info(f"  Pred: {pred_text[:50]}")


def train_crnn_model(
    dataset_path: str,
    output_model_path: str = "models/handwriting_model.keras",
    epochs: int = 100,
    batch_size: int = 32,
    log_batch_every: int = 200,
    checkpoint_dir: str = "checkpoints/handwriting_crnn",
    resume: bool = False,
    warm_start_model: Optional[str] = None
) -> bool:
    """Train CRNN model with real CTC loss on handwriting dataset"""
    
    logger.info("="*70)
    logger.info("CRNN Training with CTC Loss for Handwriting Recognition")
    logger.info("="*70)
    
    # ===== Load Dataset =====
    logger.info("\n[Step 1/4] Loading dataset...")
    dataset = HandwritingLineDataset(dataset_path)
    if not dataset.load_dataset():
        return False
    
    # ===== Build Model =====
    logger.info("\n[Step 2/4] Building CRNN model...")
    charset = dataset.charset
    model = build_crnn_model(charset)
    
    logger.info(f"Model architecture:")
    logger.info(f"  Input: (batch, 32, variable_width, 1)")
    logger.info(f"  Output: (batch, width/16, {charset.num_classes} classes incl. blank)")
    logger.info(f"  Total parameters: {model.count_params():,}")
    
    # ===== Compile Model =====
    logger.info("\n[Step 3/4] Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=ctc_loss_fn,
        metrics=[]  # CTC loss has no standard metrics
    )

    # ===== Optional Warm Start from Previous Model =====
    if warm_start_model and not resume:
        warm_start_path = Path(warm_start_model)
        if warm_start_path.exists():
            try:
                logger.info(f"Warm-starting from model: {warm_start_path}")
                old_model = keras.models.load_model(
                    str(warm_start_path),
                    custom_objects={
                        "ctc_loss_fn": ctc_loss_fn,
                        "Custom>ctc_loss_fn": ctc_loss_fn,
                    },
                    compile=False,
                )

                copied_layers = 0
                for new_layer in model.layers:
                    try:
                        old_layer = old_model.get_layer(new_layer.name)
                    except Exception:
                        continue

                    old_weights = old_layer.get_weights()
                    new_weights = new_layer.get_weights()
                    if not old_weights or not new_weights:
                        continue

                    if len(old_weights) != len(new_weights):
                        continue

                    shapes_match = all(ow.shape == nw.shape for ow, nw in zip(old_weights, new_weights))
                    if not shapes_match:
                        continue

                    new_layer.set_weights(old_weights)
                    copied_layers += 1

                logger.info(f"Warm start complete: copied weights for {copied_layers} matching layers")
            except Exception as e:
                logger.warning(f"Warm start failed ({e}); continuing with random initialization")
        else:
            logger.warning(f"Warm start model not found: {warm_start_path}")

    # ===== Resume/Checkpoint Setup =====
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    state_file = checkpoint_path / "training_state.json"

    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(checkpoint_path),
        max_to_keep=5
    )
    
    # ===== Train Model =====
    logger.info(f"\n[Step 4/4] Training for {epochs} epochs (batch_size={batch_size})...")
    logger.info("Using real CTC loss with proper sequence alignment")
    
    num_batches = len(dataset.train_images) // batch_size
    
    best_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    patience = 10

    if resume:
        if checkpoint_manager.latest_checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint_manager.latest_checkpoint}")
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

            if state_file.exists():
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)

                    start_epoch = int(state.get("next_epoch", 0))
                    best_loss = float(state.get("best_loss", float('inf')))
                    patience_counter = int(state.get("patience_counter", 0))

                    logger.info(
                        f"Resume state loaded | next_epoch={start_epoch + 1}/{epochs} | "
                        f"best_loss={best_loss:.4f} | patience_counter={patience_counter}/{patience}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load training state file {state_file}: {e}")
                    logger.warning("Continuing from checkpoint weights with epoch counter reset to 1")
                    start_epoch = 0
                    best_loss = float('inf')
                    patience_counter = 0
            else:
                logger.warning(f"State file not found at {state_file}; resuming weights only")
        else:
            logger.warning(f"No checkpoint found in {checkpoint_path}; starting fresh")

    if start_epoch >= epochs:
        logger.info(
            f"Requested epochs={epochs}, but resume state already reached next_epoch={start_epoch}. "
            "Nothing to train. Increase --epochs to continue."
        )
        return True

    def save_training_state(next_epoch: int, best: float, patience_count: int) -> None:
        state_payload = {
            "next_epoch": int(next_epoch),
            "best_loss": float(best),
            "patience_counter": int(patience_count),
            "epochs_target": int(epochs),
            "batch_size": int(batch_size)
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_payload, f, indent=2)
    
    epoch: Optional[int] = None
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0
            num_batches_actual = 0

            # Shuffle training data
            train_indices = np.random.permutation(len(dataset.train_images))

            # Mini-batch training
            for batch_start in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[batch_start:batch_start + batch_size]
                (batch_images, batch_texts), (batch_text_lens, batch_input_lens) = \
                    dataset.get_batch(batch_indices.tolist(), use_train=True)

                # Custom training loop for CTC
                with tf.GradientTape() as tape:
                    predictions = model(batch_images, training=True)
                    loss = ctc_loss_fn(
                        (batch_texts, batch_text_lens, batch_input_lens),
                        predictions
                    )

                # Backward pass
                gradients = tape.gradient(loss, model.trainable_weights)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                epoch_loss += tf.reduce_mean(loss).numpy()
                num_batches_actual += 1

                if log_batch_every > 0 and (num_batches_actual % log_batch_every == 0):
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Batch {num_batches_actual}/{num_batches} | "
                        f"Avg train loss so far: {epoch_loss / max(num_batches_actual, 1):.4f}"
                    )

            epoch_loss /= num_batches_actual

            # Validation
            val_loss = 0
            val_num_batches = 0
            val_indices = np.arange(len(dataset.val_images))

            for batch_start in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[batch_start:batch_start + batch_size]
                (val_images, val_texts), (val_text_lens, val_input_lens) = \
                    dataset.get_batch(batch_indices.tolist(), use_train=False)

                predictions = model(val_images, training=False)
                loss = ctc_loss_fn(
                    (val_texts, val_text_lens, val_input_lens),
                    predictions
                )
                val_loss += tf.reduce_mean(loss).numpy()
                val_num_batches += 1

            val_loss /= max(val_num_batches, 1)

            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs} | Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                logger.info(f"  → New best model (val_loss: {val_loss:.4f})")
                model.save(output_model_path)
            else:
                patience_counter += 1

            # Save checkpoint + state after each completed epoch.
            ckpt_path = checkpoint_manager.save(checkpoint_number=epoch + 1)
            save_training_state(
                next_epoch=epoch + 1,
                best=best_loss,
                patience_count=patience_counter
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience reached)")
                break

    except KeyboardInterrupt:
        # Save an interrupt checkpoint so user can resume later.
        interrupted_epoch = epoch if epoch is not None else start_epoch
        ckpt_path = checkpoint_manager.save(checkpoint_number=interrupted_epoch + 1)
        save_training_state(
            next_epoch=interrupted_epoch,
            best=best_loss,
            patience_count=patience_counter
        )
        logger.warning("Training interrupted by user. Resume data saved.")
        logger.warning(f"Checkpoint saved: {ckpt_path}")
        logger.warning(
            f"To continue: python train_on_real_handwriting.py --resume "
            f"--epochs {epochs} --batch-size {batch_size} --log-batch-every {log_batch_every}"
        )
        return False
    
    logger.info(f"\n✓ Training complete!")
    logger.info(f"✓ Best model saved to: {output_model_path}")
    
    return True


# ==================== Dataset Preparation Guide ====================

def print_dataset_guide():
    """Print comprehensive guide for preparing handwriting dataset"""
    guide = """
================================================================================
HANDWRITING OCR DATASET PREPARATION GUIDE
================================================================================

For FULL DOCUMENT OCR, you need:
1. Line-level image images (handwritten text lines, not full pages)
2. Text transcriptions for each line
3. Proper CTC training with this script

EXPECTED DIRECTORY STRUCTURE:
=============================

    data/
    ├── train/
    │   ├── line_001.png       (32px tall, variable width)
    │   ├── line_002.png
    │   ├── line_003.png
    │   └── ...more lines...
    ├── val/
    │   ├── line_500.png
    │   ├── line_501.png
    │   └── ...more lines...
    └── labels.txt             (transcriptions matching filenames)

LABELS.TXT FORMAT:
==================

Each line contains: filename,transcription

Example (labels.txt):
    line_001.png,The quick brown fox jumps over the lazy dog
    line_002.png,Hello world this is handwriting
    line_003.png,Another line for training
    line_004.png,Good handwriting recognition needs real data
    ...

KEY POINTS:
- PNG format (32 pixels tall recommended)
- Variable width is OK (will be padded/truncated to 512px)
- Text transcription should match exactly what's in the image
- One label per line, comma-separated

HOW TO GET/CREATE DATASETS:
===========================

OPTION 1: IAM Handwriting Database (RECOMMENDED)
    Source: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    Steps:
    1. Register for free account
    2. Download "lines.tgz" (handwritten text lines)
    3. Extract and use the line images directly
    4. Use provided metadata to create labels.txt
    
    Advantages:
    - 13,353 handwritten line images
    - Already segmented by line
    - High quality
    - Expected accuracy: 80-90% after training

OPTION 2: HWDB - Chinese Handwriting Database
    Source: http://www.nlpr.ia.ac.cn/databases/handwriting/home.html
    - Free download
    - Good for Chinese character recognition
    - Already segmented

OPTION 3: Create Your Own Dataset
    Steps:
    1. Write text on paper (standard lined paper is fine)
    2. Photograph each line clearly (camera or smartphone)
    3. Use ImageMagick/Photoshop to:
       - Crop to just the text line
       - Resize height to 32 pixels
       - Save as PNG
    4. Create labels.txt manually
    
    Tips for best results:
    - Good lighting (avoid shadows)
    - Dark pen on white background
    - Clear, natural handwriting (not tiny)
    - Variety in handwriters helps
    - At least 500-1000 samples for decent accuracy

TRAINING PROCEDURE:
===================

1. Organize dataset as shown above
2. Ensure labels.txt is correct (test with small subset first):
   python train_on_real_handwriting.py --dataset-path ./data --epochs 5

3. Full training (100 epochs is typical):
   python train_on_real_handwriting.py --dataset-path ./data --epochs 100

4. Monitor training output for convergence
   - Training loss should decrease gradually
   - Validation loss is the key metric
   - Early stopping at 10 epochs of no improvement (automatic)

EXPECTED RESULTS BY DATASET SIZE:
=================================

    10 samples:    ~5-10% accuracy (barely works)
    50 samples:    ~20-30% accuracy (better than random)
   100 samples:    ~40-50% accuracy (starting to work)
   500 samples:    ~70-80% accuracy (good)
  1000+ samples:   ~85-95% accuracy (excellent)

These numbers assume:
- Reasonable handwriting clarity
- Good image quality (not blurry)
- Proper transcriptions in labels.txt

TESTING YOUR MODEL:
===================

After training, test with:
    python test_handwriting_model.py --image your_image.png

Or use in the Flask API:
    python app.py
    # Send requests to /api/ocr/handwritten endpoint

TROUBLESHOOTING:
================

Issue: "No training images found"
Solution: Check labels.txt format (filename,text not filename|text)

Issue: Low accuracy even after training
Solution: 
  - Increase dataset size (need at least 500 samples)
  - Check image quality (should be clear, not blurry)
  - Verify labels.txt transcriptions are correct

Issue: Out of memory / slow training
Solution:
  - Reduce batch_size: --batch-size 16
  - Use fewer epochs for testing: --epochs 10

TIPS FOR BEST ACCURACY:
======================

1. Dataset Quality > Quantity
   - Clear, readable handwriting beats quantity
   - Vary handwriters if possible
   - Include numbers and punctuation in training data

2. Image Preprocessing
   - Consistent lighting
   - High contrast (dark text, white background)
   - Minimal skew/rotation
   - 32 pixels height is standard

3. Training Hyperparameters
   - Default learning_rate=0.001 works well
   - batch_size=32 is good; reduce if memory issues
   - 100 epochs is typical; use early stopping

4. For Production Deployment
   - Use ensemble of model + EasyOCR fallback (already in your app)
   - Monitor actual performance on real users' images
   - Collect misrecognized samples for retraining

================================================================================
"""
    print(guide)


# ==================== Main ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CRNN model with real CTC loss on handwriting dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See full dataset preparation guide
  python train_on_real_handwriting.py --guide
  
  # Train on dataset at ./data for 100 epochs
  python train_on_real_handwriting.py --dataset-path ./data --epochs 100
  
  # Train with smaller batch size (less memory usage)
  python train_on_real_handwriting.py --dataset-path ./data --batch-size 16
  
  # Quick test with 5 epochs
  python train_on_real_handwriting.py --dataset-path ./data --epochs 5

    # Resume from latest checkpoint
    python train_on_real_handwriting.py --resume --epochs 100
        """
    )
    
    parser.add_argument(
        "--dataset-path",
        default="./data",
        help="Path to dataset directory (default: ./data)"
    )
    parser.add_argument(
        "--output-model",
        default="models/handwriting_model.keras",
        help="Output model path (default: models/handwriting_model.keras)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--log-batch-every",
        type=int,
        default=200,
        help="Log training progress every N batches (default: 200, 0 disables batch logs)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/handwriting_crnn",
        help="Directory to save/load training checkpoints (default: checkpoints/handwriting_crnn)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in --checkpoint-dir"
    )
    parser.add_argument(
        "--warm-start-model",
        default=None,
        help="Path to an existing .keras model to copy matching layer weights from when starting a new run"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Print detailed dataset preparation guide and exit"
    )
    
    args = parser.parse_args()
    
    # Print guide if requested
    if args.guide:
        print_dataset_guide()
        sys.exit(0)
    
    # Train model
    logger.info(f"Starting training with arguments:")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Output model: {args.output_model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"  Resume: {args.resume}")
    
    success = train_crnn_model(
        dataset_path=args.dataset_path,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_batch_every=args.log_batch_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        warm_start_model=args.warm_start_model
    )
    
    sys.exit(0 if success else 1)
