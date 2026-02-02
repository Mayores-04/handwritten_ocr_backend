"""
Image to Text OCR Backend using Keras
Supports both printed and handwritten text recognition
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from typing import Any, Optional, Union
from PIL import Image, UnidentifiedImageError
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ocr_engine import OCREngine
    from utils import decode_base64_image
    logger.info("OCR modules loaded successfully")
except Exception as e:
    logger.error(f"Failed to import OCR modules: {e}")
    OCREngine = None
    decode_base64_image = None

app = Flask(__name__)

# Configure CORS with explicit origins
CORS(app, 
     origins=[
         "https://handwritten-ocr-gold.vercel.app",
         "http://localhost:3000",
         "http://127.0.0.1:3000"
     ], 
      resources={r"/api/*": {"origins": [
        "https://handwritten-ocr-gold.vercel.app"
    ]}},
     supports_credentials=True,
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Add CORS headers to all responses (fallback for error cases)
@app.after_request
def add_cors_headers(response):
    """Ensure CORS headers are always present"""
    origin = request.headers.get('Origin', '')
    allowed_origins = [
        "https://handwritten-ocr-gold.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Initialize OCR Engine (lazy loading with error handling)
ocr_engine = None

def get_ocr_engine():
    """Get OCR engine with lazy initialization"""
    global ocr_engine
    if ocr_engine is None and OCREngine is not None:
        try:
            ocr_engine = OCREngine()
            logger.info("OCR Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR Engine: {e}")
            ocr_engine = None
    return ocr_engine


# ============ Helper Functions ============

def get_image_from_request() -> Optional[Image.Image]:
    """Extract image from request (file or base64)"""
    json_data: dict[str, Any] = request.get_json(silent=True) or {}
    
    if 'image' in request.files:
        file = request.files['image']
        try:
            data = file.read()
            return Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            logger.error("Uploaded file is not a valid image or format not recognized")
            return None
    
    if 'image_base64' in json_data and decode_base64_image is not None:
        return decode_base64_image(str(json_data['image_base64']))
    
    return None


def get_mode_from_request(default: str = 'auto') -> str:
    """Extract OCR mode from request"""
    json_data: dict[str, Any] = request.get_json(silent=True) or {}
    mode = json_data.get('mode', request.form.get('mode', default))
    return str(mode) if mode else default


def format_ocr_response(result: dict[str, Any]) -> dict[str, Any]:
    """Format OCR result for API response"""
    lines: list[str] = result.get('lines', [])
    if not lines and result.get('text'):
        lines = str(result['text']).split('\n')
    
    return {
        'success': True,
        'text': result['text'],
        'confidence': result['confidence'],
        'mode_used': result.get('mode', 'unknown'),
        'word_boxes': result.get('word_boxes', []),
        'lines': lines,
        'line_count': len(lines)
    }

# ============ API Endpoints ============

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'name': 'OCR API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'ocr': '/api/ocr (POST)',
            'batch': '/api/batch-ocr (POST)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        engine = get_ocr_engine()
        ocr_status = "ready" if engine else "not_available"
        
        return jsonify({
            'status': 'healthy',
            'message': 'OCR API is running',
            'ocr_engine': ocr_status,
            'port': os.environ.get('PORT', '8000')
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/ocr', methods=['POST', 'OPTIONS'])
def extract_text():
    """Extract text from uploaded image (printed or handwritten)"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Check if OCR engine is available
        engine = get_ocr_engine()
        if not engine:
            return jsonify({
                'success': False,
                'error': 'OCR engine not available'
            }), 503
        image = get_image_from_request()
        if not image:
            return jsonify({
                'success': False,
                'error': 'No image provided. Send image file or base64 encoded image.'
            }), 400
        
        mode = get_mode_from_request()
        result = engine.recognize_text(image, mode=mode)
        
        return jsonify(format_ocr_response(result))
    
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ocr/handwritten', methods=['POST'])
def extract_handwritten() -> Union[Response, tuple[Response, int]]:
    """Specialized endpoint for handwritten text recognition"""
    try:
        engine = get_ocr_engine()
        if not engine:
            return jsonify({'success': False, 'error': 'OCR engine not available'}), 503
        
        image = get_image_from_request()
        if not image:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        result: dict[str, Any] = engine.recognize_handwritten(image)
        response = format_ocr_response(result)
        response['characters'] = result.get('characters', [])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ocr/batch', methods=['POST'])
def batch_extract() -> Union[Response, tuple[Response, int]]:
    """Process multiple images in batch"""
    try:
        engine = get_ocr_engine()
        if not engine:
            return jsonify({'success': False, 'error': 'OCR engine not available'}), 503
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'success': False, 'error': 'No images provided'}), 400
        
        results: list[dict[str, Any]] = []
        for file in files:
            try:
                data = file.read()
                image = Image.open(io.BytesIO(data))
            except UnidentifiedImageError:
                results.append({
                    'filename': file.filename,
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Invalid image file'
                })
                continue
            result: dict[str, Any] = engine.recognize_text(image, mode='auto')
            results.append({
                'filename': file.filename,
                'text': result['text'],
                'confidence': result['confidence']
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============ Run Server ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting OCR API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
