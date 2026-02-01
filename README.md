# Image to Text OCR Backend (Keras)

A Flask-based backend for Optical Character Recognition (OCR) that supports both **printed** and **handwritten** text recognition using Keras/TensorFlow deep learning models.

## 🧠 Architecture

### OCR Models Used

1. **Keras-OCR Pipeline** (for printed text)
   - **CRAFT** (Character-Region Awareness For Text detection) - CNN-based text detector
   - **CRNN** (CNN + BiLSTM + CTC) - Recognizer

2. **Custom CRNN Model** (for handwritten text)
   - **CNN layers** - Feature extraction from images
   - **Bidirectional LSTM** - Sequence modeling
   - **CTC Loss** - Connectionist Temporal Classification for training

```
Input Image → CNN Feature Extraction → BiLSTM Sequence Modeling → CTC Decoding → Text Output
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: Install Tesseract (fallback OCR)
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- Mac: `brew install tesseract`

## 🚀 Running the Server

```bash
# Development mode
python app.py

# Production mode
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Server will be available at `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```
GET /api/health
```

### Extract Text from Image
```
POST /api/ocr
Content-Type: multipart/form-data

Parameters:
- image: Image file
- mode: "printed" | "handwritten" | "auto" (default: "auto")
```

### Handwritten Text Only
```
POST /api/ocr/handwritten
Content-Type: multipart/form-data

Parameters:
- image: Image file
```

### Batch Processing
```
POST /api/ocr/batch
Content-Type: multipart/form-data

Parameters:
- images: Multiple image files
```

## 📝 Example Usage

### Python
```python
import requests

# Single image OCR
with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/ocr',
        files={'image': f},
        data={'mode': 'auto'}
    )
    print(response.json())
```

### JavaScript (Frontend)
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('mode', 'handwritten');

const response = await fetch('http://localhost:5000/api/ocr', {
    method: 'POST',
    body: formData
});
const result = await response.json();
console.log(result.text);
```

### cURL
```bash
curl -X POST http://localhost:5000/api/ocr \
  -F "image=@handwritten_note.png" \
  -F "mode=handwritten"
```

## 🏋️ Training Custom Handwriting Model

### Using IAM Handwriting Database

1. Download IAM dataset from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

2. Extract to `data/iam/` folder

3. Run training:
```bash
python train_handwriting_model.py --data ./data/iam --epochs 50
```

### Using Synthetic Data (Demo)
```bash
python train_handwriting_model.py --epochs 10
```

## 🔧 Model Architecture (CRNN)

```
Input: (32, 128, 1) grayscale image
    ↓
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
    ↓
Conv2D(128) → BatchNorm → MaxPool → Conv2D(128) → BatchNorm → MaxPool
    ↓
Conv2D(256) → BatchNorm → Dropout
    ↓
Reshape for RNN
    ↓
Bidirectional LSTM(128) → Bidirectional LSTM(64)
    ↓
Dense(num_classes + 1) with softmax
    ↓
CTC Decoding → Output Text
```

## 📁 Project Structure

```
backend/
├── app.py                    # Flask API server
├── ocr_engine.py             # OCR Engine with Keras models
├── train_handwriting_model.py # Training script
├── requirements.txt          # Python dependencies
├── models/
│   └── handwriting_model.keras  # Trained model (after training)
└── data/
    └── iam/                  # IAM dataset (optional)
```

## 🔍 Understanding Keras OCR

### Why Keras for OCR?

1. **Deep Learning Power**: CNNs excel at extracting visual features from images
2. **Sequence Modeling**: LSTMs handle variable-length text sequences
3. **CTC Loss**: Allows training without character-level segmentation
4. **Flexibility**: Easy to customize and fine-tune for specific use cases

### Key Concepts

- **CRAFT**: Detects text regions in images using CNN
- **CRNN**: Combines CNN (feature extraction) + RNN (sequence modeling)
- **CTC (Connectionist Temporal Classification)**: Loss function that handles alignment between input sequences and output labels
- **Bidirectional LSTM**: Processes sequences in both directions for better context

## 📚 References

- [keras-ocr Documentation](https://keras-ocr.readthedocs.io/)
- [CRNN Paper](https://arxiv.org/abs/1507.05717)
- [CRAFT Paper](https://arxiv.org/abs/1904.01941)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
# handwritten_ocr_backend
