FROM python:3.11-slim

# =====================
# Environment
# =====================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2

# =====================
# System Dependencies
# =====================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =====================
# App Setup
# =====================
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# =====================
# Run App (Render uses $PORT)
# =====================
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 1 \
    --timeout 300
