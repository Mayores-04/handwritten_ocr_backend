"""Test the trained character recognition model"""
import os
import numpy as np
from tensorflow import keras
import cv2

# Load the model
model_path = 'models/char_model.keras'
print(f"Loading model from {model_path}...")
model = keras.models.load_model(model_path)
print("Model loaded successfully!")

# Character classes
char_classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
print(f"Number of classes: {len(char_classes)}")

# Test with a sample image from the Img folder
test_images = [
    'Img/img001-001.png',  # First sample
    'Img/img002-001.png',  # Second sample
    'Img/img010-001.png',  # 10th sample
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\nTesting with: {img_path}")
        
        # Load and preprocess - model expects 32x32 input
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (32, 32))  # Changed from 28x28 to 32x32
        img_normalized = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_normalized, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)
        
        # Predict
        predictions = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_char = char_classes[predicted_class] if predicted_class < len(char_classes) else '?'
        
        print(f"  Predicted: '{predicted_char}' (class {predicted_class})")
        print(f"  Confidence: {confidence:.2%}")
    else:
        print(f"\nImage not found: {img_path}")

print("\n--- Model test complete! ---")
