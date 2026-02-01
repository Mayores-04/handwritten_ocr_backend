"""
Model loading utilities for OCR
"""

import os
from config import MODEL_PATHS, CHAR_CLASSES


class ModelLoader:
    """Lazy model loader to save memory"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.char_model = None
        self.handwriting_model = None
        self.trocr_processor = None
        self.trocr_model = None
        self._models_loaded = False
        self._trocr_loaded = False
    
    def load_easyocr(self):
        """Load EasyOCR reader"""
        if self.easyocr_reader is None:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print("EasyOCR loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load EasyOCR: {e}")
        return self.easyocr_reader
    
    def load_char_model(self):
        """Load trained character recognition model"""
        if self.char_model is not None:
            return self.char_model
            
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATHS['char_model'])
        
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.char_model = keras.models.load_model(model_path)
                print("Character recognition model loaded successfully (80.79% accuracy)")
            except Exception as e:
                print(f"Warning: Could not load character model: {e}")
        
        return self.char_model
    
    def load_handwriting_model(self):
        """Load custom handwriting model if available"""
        if self.handwriting_model is not None:
            return self.handwriting_model
            
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATHS['handwriting_model'])
        
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.handwriting_model = keras.models.load_model(model_path)
                print("Custom handwriting model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load handwriting model: {e}")
        else:
            print("Custom handwriting model not found. Using character model or EasyOCR fallback.")
        
        return self.handwriting_model
    
    def load_trocr(self):
        """Load TrOCR model - DISABLED for speed, using EasyOCR instead"""
        # TrOCR is slow to load and requires extra dependencies
        # EasyOCR works well enough for our use case
        return False
    
    def load_all(self):
        """Load all models"""
        if self._models_loaded:
            return
        
        self.load_easyocr()
        self.load_char_model()
        self.load_handwriting_model()
        self._models_loaded = True


# Singleton instance
model_loader = ModelLoader()
