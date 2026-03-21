# Handwritten OCR System Architecture

## Overview

This OCR system combines multiple deep learning models (Keras CRNN, Character Recognition Model, and EasyOCR) to recognize both printed and handwritten text with high accuracy and reliability.

---

## System Architecture Diagram

```
User Request (Image)
        ↓
   [Flask API]
        ↓
   [OCREngine]
        ↓
    ┌───────────┴────────────────┐
    ↓                            ↓
[Handwritten Mode]        [Printed Mode]
    ↓                            ↓
┌─────────────────┐       ┌──────────────┐
│  Handwriting    │       │ EasyOCR      │
│  Pipeline       │       │ (Single Pass)│
└────────┬────────┘       └──────────────┘
         ↓
    ┌────────────────────┐
    │ Try Keras CRNN     │
    │ (Primary)          │
    └─────────┬──────────┘
              ↓
         Success?
         /      \
       YES      NO
       /          \
     ✓         Try EasyOCR
                    ↓
              Post-process ← Conservative Fixes
                    ↓
              Return Result
```

---

## 1. Core Models

### 1.1 Keras CRNN (Convolutional Recurrent Neural Network)

**Purpose:** Recognize variable-length handwritten text sequences

**Architecture:**

```
Input (32×128 grayscale)
    ↓
[CNN Feature Extraction]
    • Conv2D (32 filters)  → MaxPool
    • Conv2D (64 filters)  → MaxPool
    • Conv2D (128 filters) → No pooling
    ↓
[Reshape] (4×16×256) → (32×1024)
    ↓
[RNN Sequence Modeling]
    • Bidirectional LSTM (128 units)
    • Dropout (0.3)
    ↓
[Output Layer]
    • Dense (63 classes: 0-9, A-Z, a-z + blank)
    ↓
Output (32×63) - logits for each timestep
```

**Model Specs:**

- **Total Parameters:** 1.29M
- **Model Size:** 14.8 MB
- **Input Shape:** (32, 128, 1) - Height, Width, Channels
- **Output Shape:** (32, 63) - Timesteps, Classes
- **Training Accuracy:** 89.88%
- **Validation Accuracy:** 89.88%

**How It Works:**

1. CNN extracts visual features from the image
2. LSTM processes features sequentially to understand context
3. Dense layer outputs probability for each character at each position
4. CTC decoder converts output to final text

**File Location:** `models/handwriting_model.keras`

---

### 1.2 Character Recognition Model

**Purpose:** Recognize individual handwritten characters (fallback for CRNN)

**Architecture:**

- Sequential Keras model
- Optimized for single-character recognition
- 62 character classes

**Model Specs:**

- **Total Parameters:** 2.56M
- **Model Size:** 9.76 MB
- **Accuracy:** 80.79%

**File Location:** `models/char_model.keras`

---

### 1.3 EasyOCR Engine

**Purpose:** Fallback OCR using pretrained deep learning model

**HOW IT WORKS:**

1. Text Detection Phase: Locates regions containing text
2. Character Recognition: Recognizes characters in detected regions
3. Post-processing: Assembles results

**Configuration:**

```python
{
    'languages': ['en'],
    'gpu': Auto-detect (torch.cuda.is_available()),
    'batch_size': 4,
    'min_size': 10,
    'decoder': 'greedy'
}
```

**Advantages:**

- Pre-trained on diverse datasets
- Handles various text orientations
- Fast inference (~1-2 seconds per image)

**Disadvantages:**

- Slower than Keras CRNN
- Heavy memory footprint
- Generic model (not optimized for your handwriting)

---

## 2. Inference Pipeline

### 2.1 Handwritten Text Recognition Flow

```
Input Image
    ↓
[preprocessing.py]
    • Convert to RGB
    • Downscale if too large (>2000px)
    ↓
[ocr_engine.py :: _run_handwriting_ocr()]
    • Upscale for clarity (2x)
    • Check Keras model availability
    ↓
    ╔════════════════════════════╗
    ║ KERAS PRIMARY ATTEMPT     ║
    ║ _keras_handwriting_recognize()
    ╚════════════════════════════╝
         ↓
    SUCCESS?
    /      \
  YES      NO
  ↓         ↓
Return  Try EasyOCR
Result    ↓
       [_easyocr_handwriting()]
            ↓
           Return
    ↓
[postprocessing.py]
    • Fix obvious OCR artifacts
    • Conservative pattern matching
    • Preserve code formatting
    ↓
Final Output
```

### 2.2 Printed Text Recognition Flow

```
Input Image
    ↓
[preprocessing.py]
    • Convert to RGB
    • Apply CLAHE contrast enhancement
    • Sharpen text
    ↓
[ocr_engine.py :: _recognize_printed()]
    • Single-pass EasyOCR
    ↓
[postprocessing.py]
    • Apply code-specific fixes
    • Spacing normalization
    ↓
Final Output
```

---

## 3. Model Preprocessing Details

### 3.1 Image Normalization

**For Keras CRNN:**

```python
Image (arbitrary size)
    ↓
Convert to BGR → Grayscale
    ↓
Resize to 32×128 pixels
    ↓
Normalize to [0, 1] (divide by 255)
    ↓
Add channel dimension → (32, 128, 1)
    ↓
Ready for CRNN inference
```

**For Printed Text (EasyOCR):**

```python
Image (arbitrary size)
    ↓
Convert to RGB (if BGR)
    ↓
Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    ↓
Apply sharpening kernel
    ↓
Ready for EasyOCR
```

### 3.2 Preprocessing Functions

**Location:** `preprocessing.py`

| Function                    | Purpose                          | Input           | Output                  |
| --------------------------- | -------------------------------- | --------------- | ----------------------- |
| `preprocess_image()`        | Basic RGB conversion & downscale | PIL/numpy image | numpy array             |
| `enhance_for_ocr()`         | CLAHE + sharpening for clarity   | RGB image       | Enhanced RGB image      |
| `upscale_for_handwriting()` | 2x upscaling for hand text       | RGB image       | Upscaled RGB            |
| `_preprocess_for_keras()`   | Convert to 32×128 grayscale      | BGR image       | (32, 128, 1) normalized |

---

## 4. Model Loading & Lazy Loading

### 4.1 Model Loader Strategy

**File:** `models.py`

```python
class ModelLoader:
    def __init__(self):
        self.easyocr_reader = None
        self.char_model = None
        self.handwriting_model = None
```

**Why Lazy Loading?**

- Reduces startup time from ~30s to ~2s
- Models loaded only when first needed
- Saves memory if models not used

### 4.2 Loading Sequence

```
App Startup
    ↓
[models.py::load_all()]
    ↓
@app.before_request
    • Load EasyOCR (if not loaded)
    • Load Keras CRNN (if not loaded)
    • Load Char model (if not loaded)
    ↓
First inference request uses preloaded models
```

**GPU Auto-Detection:**

```python
import torch
use_gpu = torch.cuda.is_available()
# EasyOCR will use GPU if available, else CPU
```

---

## 5. Inference Execution Details

### 5.1 CRNN Inference Step-by-Step

```python
# Input: handwritten image
image_input = (32, 128, 1) normalized tensor

# Forward pass through CRNN
output = model.predict(image_input)
# Output shape: (1, 32, 63) - batch_size=1, timesteps=32, classes=63

# CTC Decoding
text = _ctc_decode(output)
# Removes blanks, collapses duplicates
# Example: [5, 5, 0, 12, 12, 0, 8] → "5C8"
```

### 5.2 CTC Decoding

```python
def _ctc_decode(predictions):
    """
    Decoding variable-length sequences
    - predictions: (timesteps, num_classes) with probabilities
    - Returns: decoded text string
    """
    # 1. Get argmax for each timestep
    decoded = [argmax(p) for p in predictions]
    # Example: [5, 5, 0, 12, 12, 0, 8]

    # 2. Remove consecutive duplicates
    deduplicated = collapse_duplicates(decoded)
    # Example: [5, 0, 12, 0, 8]

    # 3. Remove blank tokens (index 0)
    final = remove_blanks(deduplicated)
    # Example: [5, 12, 8]

    # 4. Convert indices to characters
    text = ''.join([CHAR_LIST[i] for i in final])
    # Example: "5C8"

    return text
```

---

## 6. Post-Processing Pipeline

### 6.1 Conservative Fixes

**File:** `postprocessing.py`

**Purpose:** Fix ONLY obvious OCR artifacts while preserving valid code/text

**Fixed Patterns:**

```python
OBVIOUS_OCR_FIXES = {
    '|': 'l',      # pipe → l (pipe rarely appears in text)
    'rn': 'm',     # rn misread as m in some fonts
}

CODE_CRITICAL_FIXES = [
    (r'Debug\s+[_0\s]+Log\b', 'Debug.Log'),    # C# specific
    (r'\bOf\b(?!=)', '0f'),                     # C# float literal
]

CONSERVATIVE_PATTERNS = [
    (r'\s+\.', '.'),       # space before period
    (r'\s+,', ','),        # space before comma
    (r'  +', ' '),         # multiple spaces → single
    (r'\(\s+', '('),       # space after (
    (r'\s+\)', ')'),       # space before )
]
```

### 6.2 What We DON'T Fix

- Ambiguous words (could be valid)
- Dictionary corrections (too aggressive)
- Capitalization (formatting preference)
- Line breaks (structural information)

**Example:**

```
Input:  "public class GameManager MonoBehaviour"
Output: "public class GameManager MonoBehaviour"  ✓ PRESERVED
Reason: Missing ':' is not an obvious artifact

Input:  "Debug _ 0 Log ( )"
Output: "Debug.Log()"  ✓ FIXED
Reason: Clear OCR error - these symbols never appear this way
```

---

## 7. Performance Metrics

### 7.1 CRNN Model Performance

| Metric               | Value                |
| -------------------- | -------------------- |
| Training Accuracy    | 89.05%               |
| Validation Accuracy  | 89.88%               |
| Total Parameters     | 1.29M                |
| Model Size           | 14.8 MB              |
| Inference Time (CPU) | ~200-500ms per image |
| Inference Time (GPU) | ~50-100ms per image  |

### 7.2 System Performance

| Operation           | Time    | Notes                          |
| ------------------- | ------- | ------------------------------ |
| App startup (lazy)  | ~2s     | Models loaded on first request |
| First request       | ~5-8s   | Model loading + inference      |
| Subsequent requests | ~0.5-2s | Direct inference               |
| Keras handwriting   | ~400ms  | Primary path                   |
| EasyOCR fallback    | ~1-2s   | Only if Keras unavailable      |

---

## 8. Error Handling & Fallbacks

### 8.1 Graceful Degradation

```
Keras Model Available?
    ├─ YES → Use Keras CRNN
    │         ├─ Success? → Return result
    │         └─ Error? → Fall back to EasyOCR
    │
    └─ NO → Use EasyOCR directly
```

### 8.2 Exception Handling

```python
try:
    keras_result = self._keras_handwriting_recognize(image)
    if keras_result.get('text'):
        return keras_result
except Exception as e:
    print(f"Keras failed: {e}. Using EasyOCR...")

# Fallback to EasyOCR
result = self._easyocr_handwriting(image)
return result
```

---

## 9. Configuration Files

### 9.1 config.py - System Configuration

```python
EASYOCR_CONFIG = {
    'languages': ['en'],
    'gpu': False,              # Auto-detected
    'batch_size': 4,
    'min_size': 10,
    'decoder': 'greedy',
}

CHAR_CLASSES = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
# 62 characters total

MODEL_PATHS = {
    'char_model': 'models/char_model.keras',
    'handwriting_model': 'models/handwriting_model.keras',
}
```

---

## 10. Model Selection Logic

### When to Use Each Model

| Scenario          | Model      | Reason                                            |
| ----------------- | ---------- | ------------------------------------------------- |
| Handwritten text  | Keras CRNN | Trained on actual handwriting, 89% accurate, fast |
| Handwriting fails | EasyOCR    | Generic model, handles edge cases                 |
| Printed text      | EasyOCR    | Optimized for printed documents                   |
| Single character  | Char model | Specialized for isolated characters               |
| Real-time (API)   | Keras CRNN | Fast (~0.4s), low latency                         |
| Batch processing  | EasyOCR    | Can process multiple images efficiently           |

---

## 11. System Requirements

**Hardware:**

- CPU: Any modern processor (Intel/AMD)
- GPU: Optional (NVIDIA CUDA compatible for speedup)
- RAM: 8GB minimum, 16GB recommended
- Storage: 200MB for models + dependencies

**Software:**

- Python 3.12+
- TensorFlow 2.16.0
- Keras 3.0.0
- PyTorch (for EasyOCR)
- OpenCV 4.9.0+

---

## 12. Summary

```
┌─────────────────────────────────────────────────────────┐
│  Your Handwritten OCR System Architecture              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Image                                           │
│      ↓                                                  │
│  Preprocessing (normalize, resize, enhance)            │
│      ↓                                                  │
│  Model Selection                                       │
│      ├─ Handwritten? → Keras CRNN (PRIMARY)           │
│      │                ├─ GPU: ~100ms, CPU: ~400ms     │
│      │                └─ Fallback: EasyOCR if fails   │
│      └─ Printed? → EasyOCR                            │
│                    (~1-2 seconds)                      │
│      ↓                                                  │
│  Post-Processing                                       │
│      ├─ Fix obvious artifacts                        │
│      ├─ Code-specific corrections                     │
│      └─ Preserve formatting                           │
│      ↓                                                  │
│  Output Text Result                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
