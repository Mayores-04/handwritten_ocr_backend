"""API package"""

from .routes import create_api_blueprint
from .handlers import handle_ocr_request

__all__ = ['create_api_blueprint', 'handle_ocr_request']
