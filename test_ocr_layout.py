"""
Test handwritten OCR with layout preservation
"""
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Initialize reader
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR loaded!")

def ocr_with_layout(image_path):
    """OCR that preserves the multi-line layout"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Run OCR
    results = reader.readtext(img)
    
    if not results:
        print("No text detected!")
        return None
    
    # Sort by Y coordinate first (top to bottom), then X (left to right)
    results_sorted = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    
    # Group into lines based on Y position
    lines = []
    current_line_texts = []
    current_line_boxes = []
    last_y = -100
    line_threshold = 40  # pixels between lines
    
    for (bbox, text, confidence) in results_sorted:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        
        # New line detected
        if abs(y_center - last_y) > line_threshold and current_line_texts:
            # Sort current line by X position (left to right)
            line_data = list(zip(current_line_boxes, current_line_texts))
            line_data_sorted = sorted(line_data, key=lambda x: x[0][0][0])
            line_text = ' '.join([t for _, t in line_data_sorted])
            lines.append(line_text)
            current_line_texts = []
            current_line_boxes = []
        
        current_line_texts.append(text)
        current_line_boxes.append(bbox)
        last_y = y_center
    
    # Don't forget the last line
    if current_line_texts:
        line_data = list(zip(current_line_boxes, current_line_texts))
        line_data_sorted = sorted(line_data, key=lambda x: x[0][0][0])
        line_text = ' '.join([t for _, t in line_data_sorted])
        lines.append(line_text)
    
    return {
        'lines': lines,
        'raw_results': results,
        'full_text': '\n'.join(lines)
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_ocr_layout.py <image_path>")
        print("\nTrying default test image...")
        image_path = "test_sample.jpg"
    else:
        image_path = sys.argv[1]
    
    print(f"\nProcessing: {image_path}")
    print("=" * 60)
    
    result = ocr_with_layout(image_path)
    
    if result:
        print("\n📝 OCR OUTPUT (Preserving Layout):")
        print("-" * 60)
        for i, line in enumerate(result['lines'], 1):
            print(f"Line {i}: {line}")
        
        print("\n" + "=" * 60)
        print("📄 FULL TEXT:")
        print("-" * 60)
        print(result['full_text'])
        
        print("\n" + "=" * 60)
        print("🔍 DETAILED WORD DETECTION:")
        print("-" * 60)
        for (bbox, text, conf) in result['raw_results']:
            print(f'  "{text}" (confidence: {conf:.1%})')
