"""Model loading and management service - Keras/TensorFlow Primary Engine"""

import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models"
KERAS_CHAR_MODEL = MODEL_DIR / "char_model.keras"
KERAS_HANDWRITING_MODEL = MODEL_DIR / "handwriting_model.keras"


class ModelService:
    """
    Manages model loading for OCR with Keras/TensorFlow as primary engine.
    Supports lazy loading to avoid long initialization times.
    """
    
    def __init__(self):
        self._keras_char_model: Optional[Any] = None
        self._keras_handwriting_model: Optional[Any] = None
        self._models_warmed = False
        self._use_keras = False
    
    def load_keras_models(self) -> bool:
        """
        Load Keras/TensorFlow models for character and handwriting recognition.
        Returns True if at least one model loaded successfully.
        
        REQUIRED: TensorFlow/Keras is the primary OCR engine for this project.
        """
        if self._keras_char_model is not None or self._keras_handwriting_model is not None:
            return True
        
        try:
            # Try to import TensorFlow/Keras
            try:
                from tensorflow import keras
            except ImportError:
                logger.error("TensorFlow not installed. This is required for OCR.")
                logger.error("Install: pip install tensorflow>=2.16.0")
                return False
            except Exception as e:
                logger.error(f"TensorFlow import error: {str(e)}")
                return False
            
            logger.info("Attempting to load Keras models...")
            
            # Load character model if available
            if KERAS_CHAR_MODEL.exists():
                try:
                    self._keras_char_model = keras.models.load_model(str(KERAS_CHAR_MODEL))
                    logger.info("✓ Keras char_model loaded successfully")
                    self._use_keras = True
                except Exception as e:
                    logger.warning(f"Failed to load char_model: {str(e)}")
            else:
                logger.warning(f"char_model.keras not found at {KERAS_CHAR_MODEL}")
            
            # Load handwriting model if available
            if KERAS_HANDWRITING_MODEL.exists():
                try:
                    self._keras_handwriting_model = keras.models.load_model(str(KERAS_HANDWRITING_MODEL))
                    logger.info("✓ Keras handwriting_model loaded successfully")
                    self._use_keras = True
                except Exception as e:
                    logger.warning(f"Failed to load handwriting_model: {str(e)}")
            else:
                logger.warning(f"handwriting_model.keras not found at {KERAS_HANDWRITING_MODEL}")
            
            if self._use_keras:
                logger.info("✓ Keras (TensorFlow) ready - primary OCR engine")
            else:
                logger.error("No Keras models loaded! Train or provide pre-trained models.")
            
            return self._use_keras
            
        except Exception as e:
            logger.error(f"Keras loading failed: {str(e)}")
            return False
    
    def warmup_models(self) -> bool:
        """
        Warm up models on startup.
        Keras/TensorFlow is required - no fallback.
        """
        if self._models_warmed:
            return True
        
        try:
            logger.info("Warming up Keras/TensorFlow OCR models...")
            
            keras_loaded = self.load_keras_models()
            
            if keras_loaded:
                logger.info("✓ Keras/TensorFlow models loaded and warm")
                self._models_warmed = True
                return True
            else:
                logger.critical("CRITICAL: Keras/TensorFlow models failed to load!")
                logger.critical("This is the primary OCR engine for this project.")
                self._models_warmed = True
                return False
            
        except Exception as e:
            logger.error(f"Model warmup error: {str(e)}")
            self._models_warmed = True
            return False
    
    def get_keras_char_model(self) -> Optional[Any]:
        """Get the Keras character classification model"""
        if self._keras_char_model is None:
            self.load_keras_models()
        return self._keras_char_model
    
    def get_keras_handwriting_model(self) -> Optional[Any]:
        """Get the Keras handwriting recognition model"""
        if self._keras_handwriting_model is None:
            self.load_keras_models()
        return self._keras_handwriting_model
    
    def get_status(self) -> dict[str, Any]:
        """Get detailed status of loaded models"""
        return {
            'keras_char_model': self._keras_char_model is not None,
            'keras_handwriting_model': self._keras_handwriting_model is not None,
            'using_keras': self._use_keras,
            'warmed': self._models_warmed,
            'primary_engine': 'Keras/TensorFlow'
        }


# Global instance
model_service = ModelService()
