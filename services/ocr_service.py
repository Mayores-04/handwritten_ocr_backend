"""This is my OCR Service coordinator. It decides whether to use the printed or handwritten OCR service."""

import logging
from typing import Any, Union

import numpy as np
from PIL import Image

from .handwritten_ocr_service import HandwrittenOCRService
from .model_service import model_service
from .printed_ocr_service import PrintedOCRService

logger = logging.getLogger(__name__)
ImageInput = Union[Image.Image, np.ndarray]

class OCRService:
    """This class just delegates OCR to the right service based on the mode I ask for."""

    def __init__(self):
        # I set up both the printed and handwritten services here, and keep track of whether the models are warmed up.
        # I also set the confidence threshold for handwritten OCR.
        self.model_service = model_service
        self._models_warmed = False

        self.easyocr_reader = None
        self._easyocr_initialized = False

        self.keras_confidence_threshold = 0.20

        self.printed_service = PrintedOCRService()
        self.handwritten_service = HandwrittenOCRService(
            keras_confidence_threshold=self.keras_confidence_threshold
        )

        logger.info("OCR Service initialized with separated printed and handwritten services")
        logger.warning(
            "Confidence threshold set to %.1f%% (filters low-confidence predictions)",
            self.keras_confidence_threshold * 100,
        )

    def _ensure_models_warmed(self):
        # I call this to make sure my models are loaded before I try to use them.
        if self._models_warmed:
            return
        self._models_warmed = True
        self.model_service.warmup_models()

    def _ensure_easyocr_loaded(self):
        # I call this to make sure EasyOCR is loaded before I use it.
        if self._easyocr_initialized:
            return

        self._easyocr_initialized = True

        try:
            import easyocr

            logger.info("Loading EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR loaded")
        except ImportError:
            logger.error("EasyOCR not installed: pip install easyocr")
        except Exception as e:
            logger.error("EasyOCR failed to load: %s", e)

    def recognize(self, image: ImageInput, mode: str = 'printed') -> dict[str, Any]:
        # This is the main entry point. I just call the right function based on the mode.
        if image is None:
            return {'success': False, 'error': 'No image provided', 'text': '', 'mode_used': mode}

        self._ensure_models_warmed()

        try:
            if mode == 'printed':
                result = self._recognize_printed(image)
            elif mode == 'handwritten':
                result = self._recognize_handwritten(image)
            else:
                result = {'success': False, 'error': f'Unknown mode: {mode}', 'text': ''}

            if isinstance(result, dict):
                result.setdefault('mode_used', mode)
            return result
        except Exception as e:
            logger.exception("OCR failed: %s", e)
            return {'success': False, 'error': str(e), 'text': '', 'mode_used': mode}

    def _recognize_printed(self, image: ImageInput) -> dict[str, Any]:
        # This is where I run printed OCR using EasyOCR.
        self._ensure_easyocr_loaded()
        if not self.easyocr_reader:
            return {
                'success': False,
                'error': 'EasyOCR not available. Install with: pip install easyocr',
                'text': '',
                'confidence': 0.0,
                'mode': 'printed',
                'engine': 'none',
                'mode_used': 'printed',
            }

        return self.printed_service.recognize(image, self.easyocr_reader)

    def _recognize_handwritten(self, image: ImageInput) -> dict[str, Any]:
        # This is where I run handwritten OCR using my own model (if available) or EasyOCR.
        self._ensure_easyocr_loaded()
        handwriting_model = self.model_service.get_keras_handwriting_model()
        char_model = self.model_service.get_keras_char_model()
        return self.handwritten_service.recognize(
            image,
            handwriting_model=handwriting_model,
            char_model=char_model,
            easyocr_reader=self.easyocr_reader,
        )
