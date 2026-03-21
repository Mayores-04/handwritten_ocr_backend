"""
Check if Keras is actually being used in the system
"""

from models import model_loader

print("=" * 70)
print("CHECKING KERAS USAGE IN SYSTEM")
print("=" * 70)

# Try to load handwriting model
print("\n1. Loading handwriting_model.keras...")
hw_model = model_loader.load_handwriting_model()
print(f"   Result: {hw_model}")
print(f"   Type: {type(hw_model)}")

if hw_model is None:
    print("\n   [NO] Handwriting model is None")
    print("      -> System will use EasyOCR for handwriting")
else:
    print("\n   [YES] Handwriting model loaded!")
    print("      -> System will use Keras for handwriting")

# Check character model
print("\n2. Loading char_model.keras...")
char_model = model_loader.load_char_model()
print(f"   Result: {type(char_model)}")

if char_model is None:
    print("\n   [NO] Character model is None")
else:
    print("\n   [YES] Character model loaded!")

# Check EasyOCR
print("\n3. Loading EasyOCR...")
ocr_reader = model_loader.load_easyocr()
print(f"   Result: {type(ocr_reader)}")

if ocr_reader is None:
    print("\n   [NO] EasyOCR is None")
else:
    print("\n   [YES] EasyOCR loaded!")

print("\n" + "=" * 70)
print("ACTUAL SYSTEM BEHAVIOR:")
print("=" * 70)

if hw_model:
    print("[YES] Keras handwriting model: USED (primary)")
    print("[FALLBACK] EasyOCR: Used only if Keras fails")
else:
    print("[NO] Keras handwriting model: NOT USED (file missing)")
    print("[YES] EasyOCR: PRIMARY (being used now)")

print("=" * 70)

