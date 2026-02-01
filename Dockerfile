FROM python:3.11-slim

# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1

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

# Render exposes PORT automatically
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "app:app"]
