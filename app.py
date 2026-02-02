from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import logging
from typing import Any, Optional
from PIL import Image, UnidentifiedImageError

# ================== Logging ==================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== OCR Imports ==================
# We'll lazy-import OCREngine inside get_ocr_engine() to avoid
# downloading large models during container start (prevents proxy 502s)
OCREngine = None
decode_base64_image = None

# ================== App ==================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# ================== CORS (VERCEL SAFE) ==================

CORS(
    app,
    resources={
        r"/api/*": {
            # Allow requests from the frontend(s). Use '*' during testing/deploy
            # if you're getting blocked by unknown origins.
            "origins": "*"
        }
    },
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    methods=["GET", "POST", "OPTIONS"]
)


@app.after_request
def _add_cors_headers(response):
    # Ensure a basic ACAO header is present even on error responses
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response

# ================== OCR Engine ==================
# ocr_engine = None

# def get_ocr_engine():
#     global ocr_engine
#     if ocr_engine is None and OCREngine:
#         ocr_engine = OCREngine()
#     return ocr_engine
ocr_engine = None

def get_ocr_engine():
    global ocr_engine
    if ocr_engine is None:
        # Lazy import to avoid heavy work during module import time
        try:
            from ocr_engine import OCREngine as _OCREngine
            from utils import decode_base64_image as _decode_base64_image
            globals()['decode_base64_image'] = _decode_base64_image
            ocr_engine = _OCREngine()
        except Exception as e:
            logger.exception("Failed to initialize OCREngine")
            raise
    return ocr_engine

# ================== Helpers ==================
def get_image_from_request() -> Optional[Image.Image]:
    json_data: dict[str, Any] = request.get_json(silent=True) or {}

    if "image" in request.files:
        try:
            return Image.open(io.BytesIO(request.files["image"].read()))
        except UnidentifiedImageError:
            return None

    if "image_base64" in json_data and decode_base64_image:
        return decode_base64_image(json_data["image_base64"])

    return None

# ================== Routes ==================
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "OCR API",
        "status": "running",
        "health": "/api/health",
        "ocr": "/api/ocr (POST)"
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "ocr_engine": "ready" if get_ocr_engine() else "not_loaded",
        "port": os.environ.get("PORT")
    })

@app.route("/api/ocr", methods=["POST"])
def ocr():
    try:
        engine = get_ocr_engine()  # lazy load
        image = get_image_from_request()

        if not image:
            return jsonify({"success": False, "error": "No image provided"}), 400

        result = engine.recognize_text(image)

        return jsonify({
            "success": True,
            "text": result.get("text"),
            "confidence": result.get("confidence"),
            "lines": result.get("lines", [])
        })

    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({"success": False, "error": str(e)}), 500


# ================== Run (Local only) ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
