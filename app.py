"""
Image to Text OCR Backend using Keras
Supports both printed and handwritten text recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image
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
     supports_credentials=True,
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

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

def get_image_from_request():
    """Extract image from request (file or base64)"""
    json_data = request.get_json(silent=True) or {}
    
    if 'image' in request.files:
        return Image.open(request.files['image'].stream)
    
    if 'image_base64' in json_data:
        return decode_base64_image(json_data['image_base64'])
    
    return None


def get_mode_from_request(default='auto'):
    """Extract OCR mode from request"""
    json_data = request.get_json(silent=True) or {}
    return json_data.get('mode', request.form.get('mode', default))


def format_ocr_response(result):
    """Format OCR result for API response"""
    lines = result.get('lines', [])
    if not lines and result.get('text'):
        lines = result['text'].split('\n')
    
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
def extract_handwritten():
    """Specialized endpoint for handwritten text recognition"""
    try:
        image = get_image_from_request()
        if not image:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        result = ocr_engine.recognize_handwritten(image)
        response = format_ocr_response(result)
        response['characters'] = result.get('characters', [])
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ocr/batch', methods=['POST'])
def batch_extract():
    """Process multiple images in batch"""
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'success': False, 'error': 'No images provided'}), 400
        
        results = []
        for file in files:
            image = Image.open(file.stream)
            result = ocr_engine.recognize_text(image, mode='auto')
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
