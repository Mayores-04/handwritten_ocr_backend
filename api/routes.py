"""API Routes"""

import logging
from flask import Blueprint, jsonify, request
from .handlers import handle_ocr_request

logger = logging.getLogger(__name__)


def create_api_blueprint(ocr_service):
    """
    Create API blueprint with OCR routes
    
    Args:
        ocr_service: OCRService instance
    
    Returns:
        Flask Blueprint
    """
    
    api = Blueprint('api', __name__, url_prefix='/api')
    
    # ============ Health & Status ============
    
    @api.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'OCR API',
            'version': '2.0'
        })
    
    @api.route('/status', methods=['GET'])
    def status():
        """Get system status"""
        try:
            status_info = ocr_service.model_service.get_status()
            is_healthy = status_info.get('keras_ready', False)
            
            return jsonify({
                'status': 'ready' if is_healthy else 'degraded',
                'models': status_info
            })
        except Exception as e:
            logger.exception("Status check failed")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
    
    # ============ OCR Endpoints ============
    
    @api.route('/ocr', methods=['POST', 'OPTIONS'])
    def ocr():
        """
        Main OCR endpoint
        Supports both printed and handwritten text
        
        Request:
        - image: File upload (multipart/form-data)
        - image_base64: Base64 encoded image (JSON)
        - mode: 'printed' or 'handwritten' (default: 'printed')
        
        Response:
        - success: bool
        - text: Recognized text
        - confidence: float (0-1)
        - lines: List of lines
        - mode_used: Which mode was used
        - error: Error message (if failed)
        """
        
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            result = handle_ocr_request(ocr_service)
            
            status_code = 200 if result.get('success') else 400
            return jsonify(result), status_code
        
        except Exception as e:
            logger.exception("OCR endpoint exception")
            return jsonify({
                'success': False,
                'error': f"Internal error: {str(e)}",
                'text': '',
                'confidence': 0.0,
                'lines': []
            }), 500
    
    @api.route('/ocr/printed', methods=['POST', 'OPTIONS'])
    def ocr_printed():
        """
        Printed text recognition endpoint
        Optimized for documents and clear text
        """
        
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            # Force printed mode
            original_form = dict(request.form)
            request.form = {**original_form, 'mode': 'printed'}
            
            result = handle_ocr_request(ocr_service)
            
            status_code = 200 if result.get('success') else 400
            return jsonify(result), status_code
        
        except Exception as e:
            logger.exception("Printed OCR endpoint exception")
            return jsonify({
                'success': False,
                'error': f"Internal error: {str(e)}",
                'text': '',
                'confidence': 0.0,
                'lines': []
            }), 500
    
    @api.route('/ocr/handwritten', methods=['POST', 'OPTIONS'])
    def ocr_handwritten():
        """
        Handwritten text recognition endpoint
        Optimized for handwritten documents
        """
        
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            # Force handwritten mode
            original_form = dict(request.form)
            request.form = {**original_form, 'mode': 'handwritten'}
            
            result = handle_ocr_request(ocr_service)
            
            status_code = 200 if result.get('success') else 400
            return jsonify(result), status_code
        
        except Exception as e:
            logger.exception("Handwritten OCR endpoint exception")
            return jsonify({
                'success': False,
                'error': f"Internal error: {str(e)}",
                'text': '',
                'confidence': 0.0,
                'lines': []
            }), 500
    
    # ============ Info Endpoints ============
    
    @api.route('/info', methods=['GET'])
    def info():
        """Get API information"""
        return jsonify({
            'name': 'Handwritten OCR API',
            'version': '2.0',
            'description': 'Optical Character Recognition for printed and handwritten text',
            'endpoints': {
                'ocr': '/api/ocr (POST)',
                'ocr_printed': '/api/ocr/printed (POST)',
                'ocr_handwritten': '/api/ocr/handwritten (POST)',
                'health': '/api/health (GET)',
                'status': '/api/status (GET)',
                'info': '/api/info (GET)'
            },
            'supported_modes': ['printed', 'handwritten']
        })
    
    return api
