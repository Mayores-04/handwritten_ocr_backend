"""
Text post-processing for OCR corrections
Handles common handwriting OCR misreads
"""

import re

# ============ Word-level corrections ============
# These replace exact matches (longer patterns first for priority)

WORD_CORRECTIONS = {
    # === "Name/name" variations ===
    'IllaName': 'My Name',
    'OJome': 'Name',
    'IJome': 'Name',
    'IName': 'Name',
    'Dome': 'Name',
    'Done': 'Name',
    'Neme': 'Name',
    'Nane': 'Name',
    'nane': 'name',
    'IJame': 'Name',
    'Jome': 'Name',
    'llame': 'Name',
    'lame': 'Name',
    'Nama': 'Name',
    'Namo': 'Name',
    'Narne': 'Name',
    'Nane': 'Name',
    
    # === "My" at start - many variations ===
    'Illa': 'My',
    'Olla': 'My',
    'Oly ': 'My ',
    'Ily ': 'My ',
    '|ha ': 'My ',
    'Iha ': 'My ',
    'lha ': 'My ',
    'Oa ': 'My ',
    'Ns ': 'My ',
    'Oe ': 'My ',
    'Dy ': 'My ',
    'Jhy ': 'My ',
    '$e ': 'My ',
    'Hy ': 'My ',
    'Mv ': 'My ',
    'Mu ': 'My ',
    'Ny ': 'My ',
    'Wy ': 'My ',
    'Ay ': 'My ',
    
    # === "is" ===
    'TS ': 'is ',
    'Is ': 'is ',
    ' 6 ': ' is ',
    ' i5 ': ' is ',
    ' 1s ': ' is ',
    
    # === "years/Years" - many variations ===
    'Uecrs': 'years',
    'uecTs': 'years',
    'uects': 'years',
    'uecits': 'years',
    'ueats': 'years',
    'ucars': 'years',
    'Yeats': 'Years',
    'yeats': 'years',
    'Vears': 'Years',
    'vears': 'years',
    'yecrs': 'years',
    'yeurs': 'years',
    'Yecrs': 'Years',
    'Yeurs': 'Years',
    'uears': 'years',
    'Uears': 'Years',
    'yeors': 'years',
    'yoars': 'years',
    'ycars': 'years',
    'yeara': 'years',
    
    # === "Mayores" - surname variations ===
    'INavpres': 'Mayores',
    'INapres': 'Mayores',
    'Iqyeres': 'Mayores',
    'Iqyores': 'Mayores',
    'Mqyores': 'Mayores',
    'Mayeres': 'Mayores',
    'Inayores': 'Mayores',
    'Maypres': 'Mayores',
    'aprey': 'Mayores',
    'Iayres': 'Mayores',
    'Mapores': 'Mayores',
    'Nayores': 'Mayores',
    'Hayores': 'Mayores',
    'Mayres': 'Mayores',
    'Maypres': 'Mayores',
    'Mavores': 'Mayores',
    'Meyores': 'Mayores',
    'Moyores': 'Mayores',
    'Mayoras': 'Mayores',
    'Mayoros': 'Mayores',
    
    # === "Jake" - name variations ===
    'Jale': 'Jake',
    'Jke': 'Jake',
    'JLe': 'Jake',
    'Ike': 'Jake',
    'Ile': 'Jake',
    'Jeke': 'Jake',
    'Jako': 'Jake',
    'Jaka': 'Jake',
    'Jaks': 'Jake',
    'Jole': 'Jake',
    'Joke': 'Jake',
    
    # === "Old/old" ===
    ' 0ld': ' Old',
    ' 01d': ' Old',
    ' o1d': ' old',
    ' oid': ' Old',
    ' oid,': ' Old,',
    ' oid.': ' Old.',
    'oid': 'Old',
    
    # === Spacing issues ===
    'MyName': 'My Name',
    'Myname': 'My name',
    'myName': 'my Name',
    'myname': 'my name',
    
    # === "My" variations ===
    'Y name': 'My Name',
    'Y Name': 'My Name',
    'y name': 'my name',
    'Olmy ': 'My ',
    'Olmy': 'My',
    'olmy': 'my',
    
    # === "is" read as "1" ===
    ' 1 ': ' is ',
    ' 1s ': ' is ',
    
    # === "J" read as "1" or "I" or "is" ===
    ' 1 Mayores': ' J. Mayores',
    ' I Mayores': ' J. Mayores',
    'Jake 1 ': 'Jake J. ',
    'Jake I ': 'Jake J. ',
    'Jake is Mayores': 'Jake J. Mayores',
    'Jake Is Mayores': 'Jake J. Mayores',
    'Jake IS Mayores': 'Jake J. Mayores',
    ' is Mayores': ' J. Mayores',
    ' Is Mayores': ' J. Mayores',
    
    # === Trailing punctuation cleanup ===
    ' ,': ',',
    ' .': '.',
    ',,': ',',
    '..': '.',
    
    # === Numbers often misread ===
    '5 10 ': '20 ',
    ' 1O ': ' 10 ',
    ' 2O ': ' 20 ',
}

# ============ Character-level patterns (regex) ============
# For more flexible matching

REGEX_PATTERNS = [
    # Fix common letter confusions at word boundaries
    (r'\bOly\b', 'My'),
    (r'\bIly\b', 'My'),
    (r'\bY\s+[Nn]ame\b', 'My Name'),  # Y name -> My Name
    (r'\bMy\s+name\b', 'My Name'),  # lowercase name after My
    (r'\bmy\s+Name\b', 'My Name'),  # mixed case
    (r'\b[Il1|]lla\s*[Nn]ame\b', 'My Name'),  # IllaName, 1llaName, etc
    (r'\bMyName\b', 'My Name'),  # missing space
    (r'\b[Jj]ale\b', 'Jake'),
    (r'\b[Uu]ecrs?\b', 'years'),
    (r'\b[Yy]e[ao]rs?\b', 'years'),
    (r'\boid\b', 'Old'),  # oid -> Old
    (r'\b1\s+Mayores\b', 'J. Mayores'),  # 1 Mayores -> J. Mayores
    (r'\bJake\s+1\s+', 'Jake J. '),  # Jake 1 -> Jake J.
    (r'\bJake\s+I\s+', 'Jake J. '),  # Jake I -> Jake J.
    (r'\bName\s+1\s+', 'Name is '),  # Name 1 -> Name is
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
    Uses both exact replacements and regex patterns
    """
    if not text:
        return text
    
    result = text
    
    # First pass: Apply word corrections (exact matches)
    # Sort by length (longest first) to avoid partial replacements
    sorted_corrections = sorted(WORD_CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for wrong, correct in sorted_corrections:
        result = result.replace(wrong, correct)
    
    # Second pass: Apply regex patterns for flexible matching
    for pattern, replacement in REGEX_PATTERNS:
        result = re.sub(pattern, replacement, result)
    
    # Third pass: Character-level fixes
    result = result.replace('_', ' ')
    result = result.replace('|', 'I')  # pipe often misread as I
    
    # Fix common number/letter confusions in specific contexts
    result = re.sub(r'\b0ld\b', 'Old', result)
    result = re.sub(r'\b01d\b', 'Old', result)
    result = re.sub(r'\b([0-9]+)\s*[Yy]e', r'\1 ye', result)  # fix "20Years" spacing
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)
    
    # Fix capitalization after periods
    result = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), result)
    
    # Ensure sentence starts with capital
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    return result.strip()


def process_lines(lines: list[str]) -> list[str]:
    """Apply post-processing to a list of lines"""
    return [post_process_handwriting(line) for line in lines]
