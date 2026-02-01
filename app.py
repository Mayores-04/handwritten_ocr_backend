"""
Image to Text OCR Backend using Keras
Supports both printed and handwritten text recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import base64
from ocr_engine import OCREngine

app = Flask(__name__)
CORS(app)

# Initialize OCR Engine
ocr_engine = OCREngine()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'OCR API is running'
    })

@app.route('/api/ocr', methods=['POST'])
def extract_text():
    """
    Extract text from uploaded image
    Supports both printed and handwritten text
    """
    try:
        # Check if we have JSON data
        json_data = request.get_json(silent=True) or {}
        
        # Check for image in files or base64 in JSON
        has_file = 'image' in request.files
        has_base64 = 'image_base64' in json_data
        
        if not has_file and not has_base64:
            return jsonify({
                'success': False,
                'error': 'No image provided. Send image file or base64 encoded image.'
            }), 400
        
        # Handle file upload
        if has_file:
            file = request.files['image']
            image = Image.open(file.stream)
        # Handle base64 encoded image
        elif has_base64:
            image_data = base64.b64decode(json_data['image_base64'])
            image = Image.open(io.BytesIO(image_data))
        
        # Get OCR mode (printed, handwritten, or auto)
        mode = request.form.get('mode', 'auto')
        if json_data:
            mode = json_data.get('mode', mode)
        
        # Perform OCR
        result = ocr_engine.recognize_text(image, mode=mode)
        
        # Extract lines from text if not provided
        lines = result.get('lines', [])
        if not lines and result.get('text'):
            lines = result['text'].split('\n')
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'confidence': result['confidence'],
            'mode_used': result['mode'],
            'word_boxes': result.get('word_boxes', []),
            'lines': lines,
            'line_count': len(lines)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ocr/handwritten', methods=['POST'])
def extract_handwritten():
    """
    Specialized endpoint for handwritten text recognition
    Uses Keras-based CNN model
    """
    try:
        json_data = request.get_json(silent=True) or {}
        
        has_file = 'image' in request.files
        has_base64 = 'image_base64' in json_data
        
        if not has_file and not has_base64:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        if has_file:
            file = request.files['image']
            image = Image.open(file.stream)
        elif has_base64:
            image_data = base64.b64decode(json_data['image_base64'])
            image = Image.open(io.BytesIO(image_data))
        
        result = ocr_engine.recognize_handwritten(image)
        
        # Extract lines from text if not provided
        lines = result.get('lines', [])
        if not lines and result.get('text'):
            lines = result['text'].split('\n')
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'confidence': result['confidence'],
            'mode_used': result.get('mode', 'handwritten'),
            'characters': result.get('characters', []),
            'lines': lines,
            'line_count': len(lines)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ocr/batch', methods=['POST'])
def batch_extract():
    """
    Process multiple images in batch
    """
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
