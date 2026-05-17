"""API request handlers"""

import io
import re
import logging
from typing import Any, Optional, Tuple
from PIL import Image, UnidentifiedImageError
from flask import request

from utils.helpers import decode_base64_image
from utils.validators import validate_image, validate_mode

logger = logging.getLogger(__name__)


class RequestHandler:
    """Handles request parsing and validation"""
    
    @staticmethod
    def get_image_from_request() -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Extract image from request
        Supports both file upload and base64 data
        
        Returns:
            (image, error_message)
        """
        
        # Try file upload first
        if 'image' in request.files:
            try:
                file = request.files['image']
                if file and file.filename:
                    image = Image.open(io.BytesIO(file.read()))
                    
                    # Validate
                    is_valid, error = validate_image(image)
                    if not is_valid:
                        return None, error
                    
                    return image, None
            except UnidentifiedImageError:
                return None, "Invalid image file format"
            except Exception as e:
                logger.exception("File upload error")
                return None, f"File upload error: {str(e)}"
        
        # Try base64 data
        try:
            json_data = request.get_json(silent=True) or {}
            if 'image_base64' in json_data and json_data['image_base64']:
                image = decode_base64_image(json_data['image_base64'])
                
                # Validate
                is_valid, error = validate_image(image)
                if not is_valid:
                    return None, error
                
                return image, None
        except Exception as e:
            logger.exception("Base64 decode error")
            return None, f"Base64 decode error: {str(e)}"
        
        return None, "No image provided (use 'image' file or 'image_base64' in JSON)"
    
    @staticmethod
    def get_ocr_mode(forced_mode: str | None = None) -> str:
        """
        Get OCR mode from request
        Defaults to 'printed'
        """
        if forced_mode:
            mode = forced_mode.strip().lower()
            is_valid, error = validate_mode(mode)
            if not is_valid:
                logger.warning("Invalid forced mode '%s': %s", forced_mode, error)
                return "printed"
            return mode

        # Check form data first
        mode = request.form.get('mode', '').strip().lower()
        
        # Check JSON data
        if not mode:
            json_data = request.get_json(silent=True) or {}
            mode = json_data.get('mode', '').strip().lower()
        
        # Validate
        is_valid, error = validate_mode(mode)
        if not is_valid:
            logger.warning(f"Invalid mode '{mode}': {error}, using 'printed'")
            return 'printed'
        
        return mode
    
    @staticmethod
    def get_output_format() -> str:
        """
        Get output format from request
        Options: 'lines' (with line breaks), 'plain' (continuous text)
        Defaults to 'lines'
        """
        # Check form data first
        fmt = request.form.get('format', '').strip().lower()
        
        # Check JSON data
        if not fmt:
            json_data = request.get_json(silent=True) or {}
            fmt = json_data.get('format', '').strip().lower()
        
        # Validate
        if fmt in ['lines', 'plain']:
            return fmt
        
        return 'lines'  # Default


def handle_ocr_request(ocr_service: Any, forced_mode: str | None = None) -> dict[str, Any]:
    """
    Handle OCR request
    
    Args:
        ocr_service: OCRService instance
    
    Returns:
        Response dictionary
    """
    
    # Get image
    image, error = RequestHandler.get_image_from_request()
    if error:
        logger.warning(f"Image validation failed: {error}")
        return {
            'success': False,
            'error': error,
            'text': '',
            'confidence': 0.0,
            'lines': []
        }
    
    # Get mode and format
    mode = RequestHandler.get_ocr_mode(forced_mode=forced_mode)
    output_format = RequestHandler.get_output_format()
    logger.info(f"OCR request: mode={mode}, format={output_format}, image_size={image.size}")
    
    # Process
    result = ocr_service.recognize(image, mode=mode)
    
    # Apply output format
    if result.get('success'):
        if output_format == 'plain' and 'plain_text' in result:
            # Use plain text (continuous) instead of lines
            result['text'] = result['plain_text']
            result['format'] = 'plain'
        else:
            # Use lines format (default)
            result['format'] = 'lines'
        # Ensure `lines` is always a list when successful. Some handwritten
        # backends may return a single long paragraph or omit `lines`.
        lines_val = result.get('lines')
        if not isinstance(lines_val, list) or len(lines_val) == 0:
            text_val = (result.get('text') or '')
            derived = [ln.strip() for ln in re.split(r"\r?\n", text_val) if ln.strip()]
            if not derived:
                # Fallback: split on semicolons/braces for code-like samples
                parts = [p.strip() for p in re.split(r';|\}|\{', text_val) if p and p.strip()]
                if parts:
                    # re-append semicolons to look like original tokens where sensible
                    derived = [p if re.search(r'[;{}]$', p) else p + ';' for p in parts]
            result['lines'] = derived
            result['line_count'] = len(derived)
    
    # Log result
    if result.get('success'):
        logger.info(
            f"OCR success: mode={result.get('mode_used')}, format={result.get('format')}, "
            f"confidence={result.get('confidence', 0):.3f}, "
            f"text_length={len(result.get('text', ''))}"
        )
    else:
        logger.warning(f"OCR failed: {result.get('error', 'Unknown error')}")
    
    return result
