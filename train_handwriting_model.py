"""
Training script for the handwriting recognition model using Keras
This script demonstrates how to train the CRNN model on handwriting data

Dataset options:
1. IAM Handwriting Database (English)
2. MNIST (digits only)
3. EMNIST (Extended MNIST with letters)
4. Custom dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from ocr_engine import create_handwriting_model


# Character set for recognition
CHARACTERS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
CHAR_TO_NUM = {char: idx + 1 for idx, char in enumerate(CHARACTERS)}  # 0 is reserved for CTC blank
NUM_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARACTERS)}
NUM_TO_CHAR[0] = ''  # Blank token


class CTCLayer(layers.Layer):
    """
    Custom CTC Loss Layer for Keras
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    
    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        
        # Calculate input and label lengths
        input_length = tf.ones(shape=(batch_size, 1)) * tf.cast(tf.shape(y_pred)[1], tf.float32)
        label_length = tf.reduce_sum(tf.cast(y_true != 0, tf.float32), axis=1, keepdims=True)
        
        # Calculate CTC loss
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        
        return y_pred


def encode_label(text, max_length=32):
    """
    Encode text label to numerical sequence
    """
    encoded = [CHAR_TO_NUM.get(char, 0) for char in text]
    # Pad to max_length
    encoded = encoded[:max_length]
    encoded += [0] * (max_length - len(encoded))
    return encoded


def decode_prediction(prediction):
    """
    Decode CTC output to text
    """
    # Greedy decoding
    decoded = []
    prev_char = None
    
    for timestep in prediction:
        char_idx = np.argmax(timestep)
        if char_idx != 0 and char_idx != prev_char:
            decoded.append(NUM_TO_CHAR.get(char_idx, ''))
        prev_char = char_idx
    
    return ''.join(decoded)


def load_iam_dataset(data_path, max_samples=None):
    """
    Load IAM Handwriting Database
    
    Expected structure:
    data_path/
        words/
            a01/
                a01-000u/
                    a01-000u-00-00.png
                    ...
        words.txt (annotations)
    """
    images = []
    labels = []
    
    words_file = os.path.join(data_path, 'words.txt')
    words_dir = os.path.join(data_path, 'words')
    
    if not os.path.exists(words_file):
        print(f"words.txt not found at {words_file}")
        return None, None
    
    with open(words_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split(' ')
            if len(parts) < 9:
                continue
            
            word_id = parts[0]
            segmentation_result = parts[1]
            
            # Skip erroneous segmentations
            if segmentation_result == 'err':
                continue
            
            transcription = parts[-1]
            
            # Construct image path
            path_parts = word_id.split('-')
            img_path = os.path.join(
                words_dir,
                path_parts[0],
                f'{path_parts[0]}-{path_parts[1]}',
                f'{word_id}.png'
            )
            
            if os.path.exists(img_path):
                images.append(img_path)
                labels.append(transcription)
            
            if max_samples and len(images) >= max_samples:
                break
    
    return images, labels


def preprocess_image(image_path, target_width=128, target_height=32):
    """
    Preprocess image for training
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Resize maintaining aspect ratio
    h, w = img.shape
    aspect = w / h
    new_w = int(target_height * aspect)
    
    if new_w > target_width:
        new_w = target_width
    
    img = cv2.resize(img, (new_w, target_height))
    
    # Pad to target width
    if new_w < target_width:
        pad_width = target_width - new_w
        img = np.pad(img, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Invert (white text on black background)
    img = 1.0 - img
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    return img


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for training
    """
    def __init__(self, image_paths, labels, batch_size=32, img_width=128, img_height=32, max_label_length=32, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.max_label_length = max_label_length
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_labels = []
        
        for idx in batch_indexes:
            img = preprocess_image(self.image_paths[idx], self.img_width, self.img_height)
            
            if img is not None:
                batch_images.append(img)
                batch_labels.append(encode_label(self.labels[idx], self.max_label_length))
        
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def build_training_model(input_shape=(32, 128, 1), num_classes=63, max_label_length=32):
    """
    Build model with CTC loss for training
    """
    # Base model
    base_model = create_handwriting_model(input_shape, num_classes)
    
    # Add CTC layer
    labels = layers.Input(name='labels', shape=(max_label_length,), dtype='float32')
    
    ctc_layer = CTCLayer(name='ctc_loss')(labels, base_model.output)
    
    training_model = keras.Model(
        inputs=[base_model.input, labels],
        outputs=ctc_layer,
        name='training_model'
    )
    
    return training_model, base_model


def train_model(data_path, epochs=50, batch_size=32, save_path='models/handwriting_model.keras'):
    """
    Train the handwriting recognition model
    """
    print("Loading dataset...")
    image_paths, labels = load_iam_dataset(data_path, max_samples=10000)
    
    if image_paths is None:
        print("Could not load dataset. Creating synthetic data for demo...")
        # Create synthetic demo data
        image_paths, labels = create_synthetic_data()
    
    print(f"Loaded {len(image_paths)} samples")
    
    # Split data
    split_idx = int(len(image_paths) * 0.9)
    train_images, val_images = image_paths[:split_idx], image_paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create generators
    train_gen = DataGenerator(train_images, train_labels, batch_size=batch_size)
    val_gen = DataGenerator(val_images, val_labels, batch_size=batch_size, shuffle=False)
    
    # Build model
    print("Building model...")
    training_model, inference_model = build_training_model()
    
    # Compile
    training_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)
    ]
    
    # Train
    print("Training...")
    history = training_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save inference model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    inference_model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return history


def create_synthetic_data(num_samples=1000):
    """
    Create synthetic handwriting data for demo purposes
    """
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    
    print("Creating synthetic training data...")
    
    words = ['hello', 'world', 'keras', 'python', 'deep', 'learning', 
             'neural', 'network', 'image', 'text', 'recognition', 'model',
             'train', 'test', 'data', 'batch', 'epoch', 'loss', 'accuracy']
    
    image_paths = []
    labels = []
    
    temp_dir = tempfile.mkdtemp()
    
    for i in range(num_samples):
        word = np.random.choice(words)
        
        # Create image
        img = Image.new('L', (128, 32), color=255)
        draw = ImageDraw.Draw(img)
        
        # Add some randomness to simulate handwriting
        x_offset = np.random.randint(0, 10)
        y_offset = np.random.randint(5, 15)
        
        draw.text((x_offset, y_offset), word, fill=0)
        
        # Save image
        img_path = os.path.join(temp_dir, f'word_{i}.png')
        img.save(img_path)
        
        image_paths.append(img_path)
        labels.append(word)
    
    return image_paths, labels


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train handwriting recognition model')
    parser.add_argument('--data', type=str, default='./data/iam', help='Path to IAM dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='models/handwriting_model.keras', help='Output model path')
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.batch_size, args.output)
