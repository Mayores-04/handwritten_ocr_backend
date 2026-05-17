"""
Handwritten OCR API - Main Application
Simple, clean, and reliable OCR service
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from utils.env_loader import load_env_local

load_env_local()

# ============ Logging Setup ============
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ Import Services ============
from services import OCRService
from api import create_api_blueprint

# ============ Initialize App ============
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# ============ CORS Setup ============
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    methods=["GET", "POST", "OPTIONS"]
)


@app.after_request
def add_cors_headers(response):
    """Ensure CORS headers on all responses"""
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response


# ============ Initialize OCR Service ============
logger.info("Initializing OCR Service...")
ocr_service = OCRService()
logger.info("OCR Service initialized")

# ============ Register API Routes ============
api_blueprint = create_api_blueprint(ocr_service)
app.register_blueprint(api_blueprint)

# ============ Load Models at Startup ============
logger.info("Loading models at startup...")
try:
    ocr_service.model_service.warmup_models()
    logger.info("Keras model warmup finished")
except Exception as e:
    logger.error(f"Failed to load Keras models at startup: {e}")

logger.info("EasyOCR will be loaded lazily when printed OCR or fallback OCR needs it.")


# ============ Root Routes ============

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - returns API info"""
    return jsonify({
        'name': 'Handwritten OCR API',
        'version': '2.0',
        'status': 'running',
        'docs': '/api/info',
        'endpoints': {
            'health': '/api/health',
            'status': '/api/status',
            'ocr': '/api/ocr',
            'ocr_printed': '/api/ocr/printed',
            'ocr_handwritten': '/api/ocr/handwritten'
        }
    })


# ============ Error Handlers ============

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'hint': 'See /api/info for available endpoints'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({
        'success': False,
        'error': 'File size exceeds 10MB limit'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.exception(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============ Run Application ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port} (debug={debug})")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=False,     # Disable auto-reloader to avoid multiprocessing
        threaded=True           # Use threading instead of multiprocessing
    )
