# Handwritten OCR Backend

Flask API for printed and handwritten OCR.

- Printed OCR uses EasyOCR.
- Handwritten OCR tries the local Keras/TensorFlow CRNN model first, then falls back to EasyOCR when the model is missing, fails, or returns low confidence.
- Training uses the real IAM word dataset already stored in `data/full_dataset`.

## Current Important Paths

```text
handwritten_ocr_backend/
  app.py
  api/
    routes.py
    handlers.py
  services/
    model_service.py
    ocr_service.py
    handwritten_ocr_service.py
    printed_ocr_service.py
  preprocessing/
  postprocessing.py
  dataset_loader.py
  train_on_real_handwriting.py
  validate_system.py
  data/
    labels.txt
    full_dataset/
      words_new.txt
      iam_words/words/
  models/
    char_model.keras
```

The larger handwriting CRNN model is currently stored at:

```text
../large_artifacts_backup/models/handwriting_model.keras
```

`model_service.py` searches this backup path automatically. You can also set:

```powershell
$env:OCR_HANDWRITING_MODEL_PATH="C:\path\to\handwriting_model.keras"
```

## Setup

Your existing `.venv` points to a Python install that is not available in this environment. Recreate it:

```powershell
cd C:\Users\Jake\OneDrive\Desktop\School_STI\ProgLang\handwritten_ocr_backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Python 3.11 is recommended for TensorFlow/EasyOCR compatibility.

## Run Backend

```powershell
cd C:\Users\Jake\OneDrive\Desktop\School_STI\ProgLang\handwritten_ocr_backend
.\.venv\Scripts\Activate.ps1
python app.py
```

Backend URL:

```text
http://localhost:5000
```

## API

### `GET /api/health`

Returns backend status, Keras model status, and dataset detection summary.

### `GET /api/status`

Returns model loading status.

### `POST /api/ocr`

Multipart fields:

- `image`: image file
- `mode`: `printed` or `handwritten`
- `format`: `lines` or `plain`

### `POST /api/ocr/printed`

Forces printed OCR.

### `POST /api/ocr/handwritten`

Forces handwritten OCR. Keras CRNN is attempted before EasyOCR fallback.

Example:

```powershell
curl.exe -X POST http://localhost:5000/api/ocr/handwritten `
  -F "image=@data\full_dataset\iam_words\words\b04\b04-208\b04-208-02-01.png" `
  -F "format=lines"
```

## Real Dataset

The real dataset is detected from:

```text
data/full_dataset/words_new.txt
data/full_dataset/iam_words/words/
```

The older `data/labels.txt` contains converted `word_00000.png` labels, but the matching flat `data/train` and `data/val` images are not present. The fixed loader therefore uses IAM metadata and nested image paths directly.

## Train Keras CRNN

Quick smoke test:

```powershell
python train_on_real_handwriting.py --dataset-path .\data --epochs 1 --batch-size 8 --max-samples 200
```

Full training:

```powershell
python train_on_real_handwriting.py --dataset-path .\data --epochs 100 --batch-size 16
```

## Fast GPU Training

Use WSL2/Linux for modern TensorFlow GPU training. TensorFlow 2.10 was the last official native-Windows GPU build; TensorFlow 2.11+ uses WSL2 for NVIDIA CUDA GPU support.

Inside WSL2:

```powershell
wsl.exe --install
```

Restart Windows if prompted, then open Ubuntu/WSL and run:

```bash
cd /mnt/c/Users/Jake/OneDrive/Desktop/School_STI/ProgLang/handwritten_ocr_backend
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu-wsl.txt
python validate_system.py --skip-http --require-gpu
python train_on_real_handwriting.py --dataset-path ./data --epochs 100 --batch-size 8 --require-gpu --mixed-precision
```

Fast smoke test:

```bash
python train_on_real_handwriting.py --dataset-path ./data --epochs 1 --batch-size 8 --max-samples 256 --require-gpu --mixed-precision
```

GPU flags:

- `--require-gpu`: stop immediately if TensorFlow cannot see the GPU.
- `--mixed-precision`: use `mixed_float16` on GPU for faster training and lower memory use.
- `--xla`: enable TensorFlow XLA JIT. Use this only on larger GPUs; it can exceed memory on 4 GB cards.

Your Windows driver currently reports an NVIDIA GeForce GTX 1650 with 4 GB VRAM. Use `--batch-size 8` and omit `--xla` for full training.

Resume:

```powershell
python train_on_real_handwriting.py --dataset-path .\data --resume --epochs 100 --batch-size 16
```

Warm start from the backup model:

```powershell
python train_on_real_handwriting.py --dataset-path .\data --epochs 50 --warm-start-model ..\large_artifacts_backup\models\handwriting_model.keras
```

The best model is saved to:

```text
models/handwriting_model.keras
```

## Validate

Dataset and local model only:

```powershell
python validate_system.py --skip-http
```

Full system after backend and frontend are running:

```powershell
python validate_system.py
```

Specific OCR image:

```powershell
python validate_system.py --image data\full_dataset\iam_words\words\b04\b04-208\b04-208-02-01.png
```

## Troubleshooting

- `No installed Python found`: install Python 3.11 and recreate `.venv`.
- `TensorFlow/Keras could not be imported`: activate `.venv`, then reinstall `requirements.txt`.
- `TensorFlow GPU visible: FAIL`: use WSL2/Linux for TensorFlow 2.11+ GPU training, then run `nvidia-smi` and `python validate_system.py --skip-http --require-gpu`.
- `handwriting_ready: false`: set `OCR_HANDWRITING_MODEL_PATH` or place `handwriting_model.keras` in `models/`.
- `Dataset detected: 0`: check that `data/full_dataset/words_new.txt` and `data/full_dataset/iam_words/words/` exist.
- EasyOCR first-run delay: EasyOCR may download its own model files the first time printed OCR or fallback OCR runs.
