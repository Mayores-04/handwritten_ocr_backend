"""
Train CRNN model with proper CTC loss for full-document handwriting OCR

This trains word/line recognition (not full pages). The current project dataset
is IAM word-level data, and the OCR service segments uploaded handwriting into
word-like crops before CRNN inference.

Supports:
- IAM Handwriting Database (lines) - recommended: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
- HWDB - Chinese Handwriting Database
- Custom labeled line images

Supported dataset structures:
    data/
      full_dataset/
        words_new.txt
        iam_words/words/<group>/<form>/<word-id>.png

or populated flat folders:

    data/train/*.png
    data/val/*.png
    data/labels.txt    (format: filename,transcription)

Installation:
    pip install tensorflow keras pillow opencv-python numpy editdistance

Usage:
    # See dataset preparation guide
    python train_on_real_handwriting.py --guide

    # Train on your dataset
    python train_on_real_handwriting.py --dataset-path ./data --epochs 100 --batch-size 16
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
from dataset_loader import DatasetSample, discover_labeled_image_samples
from gpu_utils import configure_tensorflow_runtime
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
                # Unknown character -> space
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
    """Load real handwritten word/line images with text transcriptions."""
    
    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        seed: int = 1337,
        max_samples: Optional[int] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.test_size = test_size
        self.seed = seed
        self.max_samples = max_samples
        self.charset = CharacterSet()
        
        self.train_samples: list[DatasetSample] = []
        self.val_samples: list[DatasetSample] = []
        self.train_texts = []
        self.val_texts = []
        self.discovery_summary = {}
        self.bad_images: set[Path] = set()
        self._bad_image_warnings = 0
        self._max_bad_image_warnings = 20
        
    def load_dataset(self) -> bool:
        """Discover the real dataset and keep image paths for lazy batch loading."""
        logger.info(f"Loading dataset from {self.dataset_path}")

        discovery = discover_labeled_image_samples(
            dataset_path=self.dataset_path,
            test_size=self.test_size,
            seed=self.seed,
            max_samples=self.max_samples,
        )
        self.discovery_summary = discovery.to_dict(preview_count=5)

        for warning in discovery.warnings:
            logger.warning(warning)

        if discovery.total_samples == 0:
            logger.error("No usable labeled images found in the dataset.")
            logger.error("Expected either populated data/train + data/val folders or IAM files at:")
            logger.error("  data/full_dataset/words_new.txt")
            logger.error("  data/full_dataset/iam_words/words/")
            return False

        self.train_samples = discovery.train_samples
        self.val_samples = discovery.val_samples
        self.train_texts = [sample.text for sample in self.train_samples]
        self.val_texts = [sample.text for sample in self.val_samples]

        logger.info("Dataset source: %s", discovery.source)
        logger.info("Labels file: %s", discovery.labels_path)
        logger.info("Images root: %s", discovery.images_root)
        logger.info(
            "Samples: %d matched, %d missing images, %d skipped metadata rows",
            discovery.matched_count,
            discovery.missing_count,
            discovery.skipped_count,
        )
        logger.info(f"Train: {len(self.train_samples)}, Val: {len(self.val_samples)}")

        preview = (self.train_samples + self.val_samples)[:5]
        for sample in preview:
            logger.info("Sample: %s -> %s", sample.source_id, sample.text)

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
        samples = self.train_samples if use_train else self.val_samples
        
        batch_images = []
        batch_texts = []
        batch_text_lengths = []
        batch_input_lengths = []
        
        max_width = 512  # Maximum image width
        target_height = 32
        cnn_downsample_factor = 16  # width: /2 /2 /2 /2 across pooling blocks

        def read_image(sample: DatasetSample) -> Optional[np.ndarray]:
            if sample.image_path in self.bad_images:
                return None

            img = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.bad_images.add(sample.image_path)
                if self._bad_image_warnings < self._max_bad_image_warnings:
                    logger.warning("Skipping unreadable image: %s", sample.image_path)
                    self._bad_image_warnings += 1
                    if self._bad_image_warnings == self._max_bad_image_warnings:
                        logger.warning("Further unreadable image warnings will be suppressed.")
                return None

            return img
        
        for idx in indices:
            sample = samples[idx]
            img_raw = read_image(sample)

            resample_attempts = 0
            while img_raw is None and resample_attempts < 3 and samples:
                resample_attempts += 1
                sample = samples[int(np.random.randint(0, len(samples)))]
                img_raw = read_image(sample)

            if img_raw is None:
                continue

            # Preprocess image
            img = self.preprocess_image(img_raw, target_height)
            
            # Truncate/pad to max width
            h, w = img.shape  
            if w > max_width:
                img = img[:, :max_width]
            else:
                # Pad right with white (1.0)
                padded = np.ones((h, max_width), dtype=np.float32)
                padded[:, :w] = img
                img = padded
            
            batch_images.append(img)
            
            # Encode text
            encoded = self.charset.encode_text(sample.text)
            batch_texts.append(encoded)
            batch_text_lengths.append(len(encoded))
            
            # Images are padded to max_width, so the model emits max_width/16
            # CTC timesteps. Passing the shorter visual width can make labels
            # longer than the input sequence and break real IAM word samples.
            input_len = max(1, max_width // cnn_downsample_factor)
            batch_input_lengths.append(input_len)
        
        if not batch_images:
            raise ValueError("No readable images found in the batch. Check dataset file integrity.")

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
    Keras CTC loss for variable-length sequences.
    
    Args:
        y_true: Tuple of (text_indices, text_lengths, input_lengths)
        y_pred: Softmax model predictions (batch, time_steps, num_classes)
    """
    text_indices, text_lengths, input_lengths = y_true
    y_pred = tf.cast(y_pred, tf.float32)
    labels = tf.cast(text_indices, tf.int32)
    label_lengths = tf.cast(tf.reshape(text_lengths, [-1]), tf.int32)
    logit_lengths = tf.cast(tf.reshape(input_lengths, [-1]), tf.int32)

    batch_size = tf.shape(labels)[0]
    max_label_length = tf.shape(labels)[1]
    mask = tf.sequence_mask(label_lengths, maxlen=max_label_length)
    sparse_indices = tf.where(mask)
    sparse_values = tf.gather_nd(labels, sparse_indices)
    sparse_labels = tf.SparseTensor(
        indices=tf.cast(sparse_indices, tf.int64),
        values=sparse_values,
        dense_shape=tf.cast([batch_size, max_label_length], tf.int64),
    )

    # The model outputs softmax probabilities for inference. log(probabilities)
    # works as CTC logits because log_softmax(log(probabilities)) is equivalent.
    logits = tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
    num_classes = y_pred.shape[-1]
    if num_classes is None:
        raise ValueError("CTC loss requires a statically known number of output classes.")

    return tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=logits,
        label_length=label_lengths,
        logit_length=logit_lengths,
        logits_time_major=False,
        blank_index=int(num_classes) - 1,
    )


class CTCCallback(keras.callbacks.Callback):
    """Custom callback for CTC-based training"""
    
    def __init__(self, dataset: HandwritingLineDataset, charset: CharacterSet):
        super().__init__()
        self.dataset = dataset
        self.charset = charset
    
    def on_epoch_end(self, epoch, logs=None):
        """Sample validation predictions every 10 epochs"""
        if epoch % 10 == 0 and epoch > 0 and self.dataset.val_samples:
            logger.info(f"\n[Epoch {epoch}] Sample predictions:")
            
            # Sample a few validation images
            val_indices = np.random.choice(len(self.dataset.val_samples), 
                                          min(3, len(self.dataset.val_samples)), 
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
    batch_size: int = 16,
    log_batch_every: int = 200,
    checkpoint_dir: str = "checkpoints/handwriting_crnn",
    resume: bool = False,
    warm_start_model: Optional[str] = None,
    test_size: float = 0.2,
    seed: int = 1337,
    max_samples: Optional[int] = None,
    require_gpu: bool = False,
    mixed_precision: bool = False,
    xla: bool = False,
) -> bool:
    """Train CRNN model with real CTC loss on handwriting dataset"""
    
    logger.info("="*70)
    logger.info("CRNN Training with CTC Loss for Handwriting Recognition")
    logger.info("="*70)

    try:
        runtime_info = configure_tensorflow_runtime(
            tf,
            require_gpu=require_gpu,
            enable_mixed_precision=mixed_precision,
            enable_xla=xla,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        return False
    logger.info("Runtime: %s", json.dumps(runtime_info.to_dict(), indent=2))
    
    # ===== Load Dataset =====
    logger.info("\n[Step 1/4] Loading dataset...")
    dataset = HandwritingLineDataset(
        dataset_path,
        test_size=test_size,
        seed=seed,
        max_samples=max_samples,
    )
    if not dataset.load_dataset():
        return False
    
    # ===== Build Model =====
    logger.info("\n[Step 2/4] Building CRNN model...")
    charset = dataset.charset
    model = build_crnn_model(charset)
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    
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
    
    num_batches = max(1, (len(dataset.train_samples) + batch_size - 1) // batch_size)
    
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

    @tf.function(reduce_retracing=True)
    def train_step(batch_images, batch_texts, batch_text_lens, batch_input_lens):
        with tf.GradientTape() as tape:
            predictions = model(batch_images, training=True)
            loss = ctc_loss_fn(
                (batch_texts, batch_text_lens, batch_input_lens),
                predictions,
            )
            mean_loss = tf.reduce_mean(loss)

        gradients = tape.gradient(mean_loss, model.trainable_weights)
        grads_and_vars = [
            (grad, var)
            for grad, var in zip(gradients, model.trainable_weights)
            if grad is not None
        ]
        model.optimizer.apply_gradients(grads_and_vars)
        return mean_loss

    @tf.function(reduce_retracing=True)
    def validation_step(val_images, val_texts, val_text_lens, val_input_lens):
        predictions = model(val_images, training=False)
        loss = ctc_loss_fn(
            (val_texts, val_text_lens, val_input_lens),
            predictions,
        )
        return tf.reduce_mean(loss)

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
            train_indices = np.random.permutation(len(dataset.train_samples))

            # Mini-batch training
            for batch_start in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[batch_start:batch_start + batch_size]
                (batch_images, batch_texts), (batch_text_lens, batch_input_lens) = \
                    dataset.get_batch(batch_indices.tolist(), use_train=True)

                loss = train_step(
                    batch_images,
                    batch_texts,
                    batch_text_lens,
                    batch_input_lens,
                )

                epoch_loss += float(loss.numpy())
                num_batches_actual += 1

                if log_batch_every > 0 and (num_batches_actual % log_batch_every == 0):
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Batch {num_batches_actual}/{num_batches} | "
                        f"Avg train loss so far: {epoch_loss / max(num_batches_actual, 1):.4f}"
                    )

            epoch_loss /= max(num_batches_actual, 1)

            # Validation
            val_loss = 0
            val_num_batches = 0
            val_indices = np.arange(len(dataset.val_samples))

            for batch_start in range(0, len(val_indices), batch_size):
                batch_indices = val_indices[batch_start:batch_start + batch_size]
                (val_images, val_texts), (val_text_lens, val_input_lens) = \
                    dataset.get_batch(batch_indices.tolist(), use_train=False)

                loss = validation_step(
                    val_images,
                    val_texts,
                    val_text_lens,
                    val_input_lens,
                )
                val_loss += float(loss.numpy())
                val_num_batches += 1

            val_loss /= max(val_num_batches, 1)

            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs} | Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                logger.info(f"  -> New best model (val_loss: {val_loss:.4f})")
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
    
    logger.info("\nTraining complete!")
    logger.info(f"Best model saved to: {output_model_path}")
    
    return True


# ==================== Dataset Preparation Guide ====================

def print_dataset_guide():
    """Print comprehensive guide for preparing handwriting dataset"""
    guide = """
================================================================================
HANDWRITING OCR DATASET PREPARATION GUIDE
================================================================================

For FULL DOCUMENT OCR, this project trains on word/line crops, then the API
segments uploaded handwriting before CRNN inference.

The current project already includes IAM word-level data at:
    data/full_dataset/words_new.txt
    data/full_dataset/iam_words/words/

You need:
1. Word-level or line-level handwritten image crops
2. Text transcriptions for each crop
3. Proper CTC training with this script

EXPECTED DIRECTORY STRUCTURE:
=============================

Preferred current project structure:

    data/
      full_dataset/
        words_new.txt
        iam_words/words/<group>/<form>/<word-id>.png

Also supported:

    data/
    train/
      line_001.png       (32px tall, variable width)
      line_002.png
      line_003.png
      ...more crops...
    val/
      line_500.png
      line_501.png
      ...more crops...
    labels.txt             (transcriptions matching filenames)

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
   - batch_size=16 is a safer default for 4 GB GPUs; use 8 if memory is tight
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
        default=16,
        help="Training batch size (default: 16; use 8 for low VRAM, 32+ for larger GPUs)"
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
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split for discovered IAM samples (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic train/validation split (default: 1337)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick training smoke tests"
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail immediately if TensorFlow cannot see a GPU"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable TensorFlow mixed_float16 policy when a GPU is available"
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="Enable TensorFlow XLA JIT compilation"
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
    logger.info(f"  Validation split: {args.test_size}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Max samples: {args.max_samples or 'all'}")
    logger.info(f"  Require GPU: {args.require_gpu}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  XLA: {args.xla}")
    
    success = train_crnn_model(
        dataset_path=args.dataset_path,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_batch_every=args.log_batch_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        warm_start_model=args.warm_start_model,
        test_size=args.test_size,
        seed=args.seed,
        max_samples=args.max_samples,
        require_gpu=args.require_gpu,
        mixed_precision=args.mixed_precision,
        xla=args.xla,
    )
    
    sys.exit(0 if success else 1)
