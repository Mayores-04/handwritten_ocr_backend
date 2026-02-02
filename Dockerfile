FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Disable GPU usage (Render has no GPU)
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies for OpenCV & OCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    tesseract-ocr \
    curl \
    libgthread-2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render provides PORT env var)
EXPOSE 8000

# Start the application - use shell form for $PORT expansion
CMD exec gunicorn --bind "0.0.0.0:${PORT:-8000}" --workers 1 --timeout 300 --preload app:app
