"""This is my model loading and management service. I use Keras/TensorFlow as my primary OCR engine."""

# I keep the model paths here so I can easily change them if I move things around.

import logging
from typing import Optional, Any
from pathlib import Path

from train_on_real_handwriting import ctc_loss_fn

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models"
KERAS_CHAR_MODEL = MODEL_DIR / "char_model.keras"
KERAS_HANDWRITING_MODEL = MODEL_DIR / "handwriting_model.keras"


class ModelService:
    """
    I use this class to manage loading my OCR models. I use lazy loading so I don't have to wait forever on startup.
    """
    
    def __init__(self):
        self._keras_char_model: Optional[Any] = None
        self._keras_handwriting_model: Optional[Any] = None
        self._models_warmed = False
        self._use_keras = False
    
    def load_keras_models(self) -> bool:
        """
        This is where I load my Keras/TensorFlow models for character and handwriting recognition.
        If TensorFlow isn't installed, I log an error and tell myself how to fix it.
        """
        if self._keras_char_model is not None or self._keras_handwriting_model is not None:
            return True
        
        try:
            # Try to import TensorFlow/Keras
            try:
                import keras
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
                    self._keras_handwriting_model = keras.models.load_model(
                        str(KERAS_HANDWRITING_MODEL),
                        custom_objects={
                            "ctc_loss_fn": ctc_loss_fn,
                            "Custom>ctc_loss_fn": ctc_loss_fn,
                        },
                        compile=False,
                    )
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
        I use this to warm up my models on startup. No fallback here—Keras/TensorFlow is required.
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
        """I use this to get the Keras character classification model."""
        if self._keras_char_model is None:
            self.load_keras_models()
        return self._keras_char_model
    
    def get_keras_handwriting_model(self) -> Optional[Any]:
        """I use this to get the Keras handwriting recognition model."""
        if self._keras_handwriting_model is None:
            self.load_keras_models()
        return self._keras_handwriting_model
    
    def get_status(self) -> dict[str, Any]:
        """This gives me a quick status of which models are loaded and what engine I'm using."""
        return {
            'keras_char_model': self._keras_char_model is not None,
            'keras_handwriting_model': self._keras_handwriting_model is not None,
            'using_keras': self._use_keras,
            'warmed': self._models_warmed,
            'primary_engine': 'Keras/TensorFlow'
        }


# Global instance
model_service = ModelService()
