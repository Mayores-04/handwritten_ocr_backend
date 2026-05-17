"""Services package with lazy exports.

Keeping these imports lazy lets diagnostics import model_service without also
requiring OpenCV/EasyOCR-related modules to import successfully.
"""

__all__ = [
    "OCRService",
    "ModelService",
    "PrintedOCRService",
    "HandwrittenOCRService",
]


def __getattr__(name):
    if name == "OCRService":
        from .ocr_service import OCRService

        return OCRService
    if name == "ModelService":
        from .model_service import ModelService

        return ModelService
    if name == "PrintedOCRService":
        from .printed_ocr_service import PrintedOCRService

        return PrintedOCRService
    if name == "HandwrittenOCRService":
        from .handwritten_ocr_service import HandwrittenOCRService

        return HandwrittenOCRService

    raise AttributeError(f"module 'services' has no attribute {name!r}")
