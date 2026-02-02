FROM python:3.11-slim

# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1

# Disable GPU usage (Render has no GPU)
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies for OpenCV & OCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose port
EXPOSE 8000

# Start with Gunicorn using config file
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
