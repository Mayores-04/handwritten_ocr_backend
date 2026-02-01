"""
Quick test for handwritten image OCR with layout preservation
"""
import requests
import sys
from PIL import Image
import io

def test_ocr_api(image_path):
    """Test the OCR API with an image file"""
    print(f"\n📷 Testing OCR on: {image_path}")
    print("=" * 60)
    
    # Read the image
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'mode': 'handwritten'}
        
        # Send to API
        response = requests.post(
            'http://localhost:5000/api/ocr/handwritten',
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get('success'):
            print("\n✅ OCR SUCCESS!")
            print("-" * 60)
            
            print("\n📝 TEXT OUTPUT (with layout preserved):")
            print("-" * 60)
            print(result['text'])
            
            if 'lines' in result:
                print("\n📋 LINE BY LINE:")
                print("-" * 60)
                for i, line in enumerate(result['lines'], 1):
                    print(f"  Line {i}: {line}")
            
            print("\n📊 STATS:")
            print("-" * 60)
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print(f"  Mode: {result.get('mode_used', 'N/A')}")
            if 'line_count' in result:
                print(f"  Lines: {result['line_count']}")
            print(f"  Characters: {len(result['text'])}")
            print(f"  Words: {len(result['text'].split())}")
        else:
            print(f"❌ OCR failed: {result.get('error')}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path>")
        print("\nExample: python test_api.py my_handwriting.jpg")
    else:
        test_ocr_api(sys.argv[1])
