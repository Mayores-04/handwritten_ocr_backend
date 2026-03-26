"""OCR Service coordinator: routes to printed or handwritten service."""

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
    """Coordinator service that delegates OCR by mode."""

    def __init__(self):
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
        if self._models_warmed:
            return
        self._models_warmed = True
        self.model_service.warmup_models()

    def _ensure_easyocr_loaded(self):
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
        char_model = self.model_service.get_keras_char_model()
        return self.handwritten_service.recognize(image, char_model)
