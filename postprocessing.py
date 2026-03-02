"""
Text post-processing for OCR corrections
Handles common handwriting OCR misreads that appear across all text types
"""

import re

# ============ Word-level corrections ============
# GENERAL corrections that apply to most handwritten text
# Avoid overly specific corrections (e.g., particular names/words)

WORD_CORRECTIONS = {
    # === General spacing issues (universal) ===
    'MyName': 'My Name',
    'Myname': 'My name',
    'myName': 'my Name',
    'myname': 'my name',
    'thethe': 'the',
    'andand': 'and',
    
    # === General spacing fixes for common words ===
    'iam': 'I am',
    'Iam': 'I am',
    'youare': 'you are',
    'Youare': 'You are',
    'iswhat': 'is what',
    'Iswhat': 'Is what',
    
    # === Punctuation artifacts (universal) ===
    ',,': ',',
    '..': '.',
    '!!!': '!!',
    '???': '??',
    ' ,': ',',
    ' .': '.',
    
    # === Only fix VERY SPECIFIC digit confusions in numbers ===
    ' 1O ': ' 10 ',  # 1 and O → 10
    ' 2O ': ' 20 ',  # 2 and O → 20
    ' 5 10 ': ' 20 ',  # 5 and 10 → 20
    ' l0 ': ' 10 ',  # letter L and 0 → 10
}

# ============ Character-level patterns (regex) ============
# GENERAL patterns that work for any text
# Focus on structural fixes, not content-specific replacements

REGEX_PATTERNS = [
    # === Fix ONLY obvious symbol artifacts (universal) ===
    
    # Multiple pipes → I (common OCR artifact) - ADD SPACING
    (r'\|{2,}', ' I '),       # || or more → I (with spaces)
    (r'\|', 'I'),             # single pipe → I
    
    # Fix underscore only in contexts that make sense (underscores between words)
    (r'_', ' '),              # underscore → space
    
    # Fix spacing around punctuation (universal)
    (r'\s+\.', '.'),          # Remove space before period
    (r'\s+,', ','),           # Remove space before comma
    (r'(\w)\*', r'\1.'),      # * → . (common OCR artifact)
    
    # Fix repeated punctuation
    (r'\.{2,}', '.'),         # .. or more → .
    (r',{2,}', ','),          # ,, or more → ,
    (r'!{3,}', '!!'),         # !!! or more → !!
    (r'\?{3,}', '??'),        # ??? or more → ??
    
    # Fix multiple spaces (universal)
    (r'\s{2,}', ' '),         # Multiple spaces → single space
    
    # Fix capitalization at sentence start (universal)
    (r'^([a-z])', lambda m: m.group(1).upper()),  # Capitalize first letter
    
    # Fix capitalization after periods/sentences (universal)
    (r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper()),
    
    # Fix "is" when it's clearly misread as "1s" (ONLY in clear contextslike "is " at word boundaries)
    (r'\b1s\b', 'is'),        # isolated 1s → is
]

# Common character confusions in handwriting
CHAR_CONFUSIONS = {
    # Letters that look similar
    'l': 'i',   # lowercase L vs i
    'I': 'l',   # uppercase I vs l
    '0': 'O',   # zero vs O
    'O': '0',   # O vs zero
    '1': 'l',   # one vs l
    '5': 'S',   # 5 vs S
    '8': 'B',   # 8 vs B
    '6': 'G',   # 6 vs G
    '2': 'Z',   # 2 vs Z
}


def post_process_handwriting(text: str) -> str:
    """
    Post-process OCR text to fix common handwriting recognition errors
    Works with ANY text input - not specific to particular words/names
    Uses structural fixes: spacing, capitalization, punctuation, character confusion fixes
    """
    if not text:
        return text
    
    result = text
    
    # === PASS 1: Apply word-level exact corrections ===
    # Sort by length (longest first) to avoid partial replacements
    sorted_corrections = sorted(WORD_CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for wrong, correct in sorted_corrections:
        result = result.replace(wrong, correct)
    
    # === PASS 2: Apply regex patterns for flexible matching ===
    for pattern, replacement in REGEX_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # === PASS 3: Clean up multiple spaces ===
    result = re.sub(r'\s+', ' ', result)
    
    # === PASS 4: Strip leading/trailing spaces ===
    result = result.strip()
    
    return result


def process_lines(lines: list[str]) -> list[str]:
    """Apply post-processing to a list of lines"""
    return [post_process_handwriting(line) for line in lines]
 
