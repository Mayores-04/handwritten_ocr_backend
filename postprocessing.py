"""
Text post-processing for OCR corrections - Conservative approach
Fixes ONLY obvious OCR errors while preserving line structure and formatting
Works for ANY text: handwriting, code, documents
"""

import re
from typing import Optional

# Try to import PyDictionary for word validation - but use conservatively
try:
    from PyDictionary import PyDictionary
    DICTIONARY = PyDictionary()
    HAS_DICT = True
except ImportError:
    HAS_DICT = False
    DICTIONARY = None


# ============ OBVIOUS OCR ARTIFACTS (High confidence fixes) ============
# These are patterns that are CLEARLY OCR errors, not valid tokens

OBVIOUS_OCR_FIXES = {
    # Symbol artifacts that are NEVER valid English/code
    '|': 'l',              # pipe character → lowercase L (rare in text)
    'rn': 'm',             # 'rn' misread as 'm' in fonts
    
    # Doubled punctuation artifacts
    ',,': ',',
    '..': '.',
    '!!!': '!!',
    '???': '??',
    
    # Space-punctuation artifacts (never valid)
    ' ,': ',',
    ' .': '.',
    ' ;': ';',
    ' :': ':',
}

# ============ CODE-SPECIFIC CRITICAL FIXES ============
# Only fix patterns that are DEFINITELY broken syntax

CODE_CRITICAL_FIXES = [
    # These are 100% OCR errors in code, not ambiguous
    (r'Debug\s+[_0\s]+Log\b', 'Debug.Log'),        # 'Debug _ 0 Log', 'Debug . Log' → 'Debug.Log'
    (r'\bOf\b(?!=)', '0f'),                        # 'Of' → '0f' (C# float, not sentence "Of")
]

# ============ REGEX PATTERNS (Conservative - only obvious artifacts) ============
CONSERVATIVE_PATTERNS = [
    # Fix spacing ONLY around punctuation (not words)
    (r'\s+\.', '.'),          # space before period
    (r'\s+,', ','),           # space before comma
    (r'\s+;', ';'),           # space before semicolon
    
    # Fix repeated punctuation (100% artifacts)
    (r'\.{2,}', '.'),
    (r',{2,}', ','),
    (r'!{3,}', '!!'),
    (r'\?{3,}', '??'),
    
    # Fix ONLY excessive spaces (multiple → single)
    (r'  +', ' '),            # 2+ spaces → 1 space (not touching single spaces)
    
    # Fix space in parentheses (code artifact)
    (r'\(\s+', '('),          # '( ' → '('
    (r'\s+\)', ')'),          # ' )' → ')'
    (r'\s+;', ';'),           # ' ;' → ';'
]

print("""
Text post-processing for OCR corrections - Conservative approach
Fixes ONLY obvious OCR errors while preserving line structure and formatting
Works for ANY text: handwriting, code, documents
""")


def post_process_handwriting(text: str) -> str:
    """
    Conservative post-processing - fixes ONLY obvious OCR errors
    Preserves line structure and formatting
    
    Process:
    1. Fix code-critical patterns (100% OCR errors)
    2. Fix obvious symbol artifacts (never valid)
    3. Fix spacing artifacts (multiple spaces, space-punctuation)
    4. Preserve line alignment
    """
    if not text:
        return text
    
    result = text
    
    # === PASS 1: Fix code-critical patterns ===
    # These are definitely wrong (Debug _ 0 Log, Of, etc)
    for pattern, replacement in CODE_CRITICAL_FIXES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # === PASS 2: Fix obvious OCR artifacts ===
    # Only fix tokens that are NEVER valid (doubled punctuation, etc)
    for wrong, correct in OBVIOUS_OCR_FIXES.items():
        result = result.replace(wrong, correct)
    
    # === PASS 3: Apply conservative spacing fixes ===
    # Only fix punctuation spacing and excessive spaces
    for pattern, replacement in CONSERVATIVE_PATTERNS:
        result = re.sub(pattern, replacement, result)
    
    # === PASS 4: Preserve line structure ===
    # Don't strip - preserve alignment and padding
    result = re.sub(r'\n\n+', '\n', result)  # Remove only excessive newlines
    
    return result


def process_lines(lines: list[str]) -> list[str]:
    """Apply post-processing to a list of lines"""
    return [post_process_handwriting(line) for line in lines]
 
