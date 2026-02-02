#!/usr/bin/env python3
"""
Startup script for production deployment on Render
"""
import os
import sys
from app import app

def main():
    """Start the application with proper configuration"""
    port = int(os.environ.get('PORT', 8000))
    host = '0.0.0.0'
    
    print(f"Starting OCR API on {host}:{port}")
    print(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
    
    # In production, use gunicorn (handled by Dockerfile)
    # This script is for debugging/local testing
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(host=host, port=port, debug=True)
    else:
        print("Production mode - should be started with gunicorn")
        return

if __name__ == '__main__':
    main()