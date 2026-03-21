FROM python:3.11-slim

# =====================
# Environment (Critical for ML models)
# =====================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PIP_NO_CACHE_DIR=1

# =====================
# System Dependencies (Required by OpenCV, EasyOCR, Keras)
# =====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =====================
# Python Setup
# =====================
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application
COPY . .

# =====================
# Health Check (Render will monitor this)
# =====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# =====================
# Run App (Render uses $PORT environment variable)
# =====================
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
