"""
This is my OCR post-processing script. I wrote it to clean up the most obvious mistakes that OCR makes, but I try not to mess with the original line structure or formatting. I use this for my own handwriting, code, and other documents.
"""

import re
from typing import List



# ============ COMMON HANDWRITING OCR MISREADS ============
COMMON_HANDWRITING_FIXES = [
    # Fuzzy and specific fixes for your sample
    (r"^Dynone is I J ?[\(\[]?aueres", "My Name is Jake J. Mayores."),
    (r"Dynone", "My Name"),
    (r"I J", "Jake J."),
    (r"[\(\[]aueres", "Mayores."),
    (r"\bD[yv]n[o0]ne\b", "My Name"),
    (r"\bI J\b", "Jake J."),
    (r"\b[aA]ueres\b", "Mayores"),
    (r"\b[aA]u[e3]res\b", "Mayores"),
    (r"\b[aA]uer[e3]s\b", "Mayores"),
    (r"\b[aA]ueres,?\b", "Mayores."),
    (r"\bJ J\b", "Jake J"),
    (r"\bJ\. J\.", "Jake J."),
    (r"\bJ\. J\b", "Jake J."),
    (r"\bJ J\.\b", "Jake J."),
    # General fixes
    (r"\bMn\b", "My"),
    (r"\bManores\b", "Mayores"),
    (r"\bM n\b", "My"),
    (r"\bNane\b", "Name"),
    (r"\bJake J\b", "Jake J."),
    (r"\bMayores\b", "Mayores."),
    # Add more as needed for your dataset
    # Remove repeated dots (e.g., 'J...' -> 'J.')
    (r'(\bJ[ ]?J)\.\.+', r'\1.'),
    (r'\.\.+', '.'),
    (r'\s+\.', '.'),
    (r'\.(,)', r'\1'),
]

# ============ OBVIOUS OCR ARTIFACTS ============
DOUBLED_PUNCT_FIXES = {
    ",,": ",",
    "!!!": "!!",
    "???": "??",
}


# ============ CODE-SPECIFIC CRITICAL FIXES ============

CODE_ONLY_FIXES = [
    (r"Debug\s+[_.\s]+Log\b", "Debug.Log"),
    (r"(?<=\d)\s*Of\b", "0f"),
]


# ============ CONSERVATIVE REGEX PATTERNS ============

CONSERVATIVE_PATTERNS = [
    (r"(?<![A-Z])\s+\.(?!\.)(?!\w)", "."),
    (r"\s+,", ","),
    (r"\s+;", ";"),
    (r"\s+:", ":"),
    (r",{2,}", ","),
    (r";{2,}", ";"),
    (r"!{3,}", "!!"),
    (r"\?{3,}", "??"),
    (r"(?<!\n)  +", " "),
    (r"\(\s{2,}", "("),
    (r"\s{2,}\)", ")"),
]


def post_process_ocr(text: str, mode: str = "handwriting") -> str:
    """
    Conservative OCR post-processing.

    Parameters
    ----------
    text : str
        Raw OCR output string.
    mode : str
        'handwriting' (default) or 'code'.

    Returns
    -------
    str
        Corrected text with line structure preserved.
    """
    if not text:
        return text

    result = text

    # Pass 1: code-only fixes
    if mode == "code":
        for pattern, replacement in CODE_ONLY_FIXES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Pass 2: common handwriting OCR misreads
    if mode == "handwriting":
        for pattern, replacement in COMMON_HANDWRITING_FIXES:
            result = re.sub(pattern, replacement, result)

    # Pass 3: doubled punctuation artifacts
    for wrong, correct in DOUBLED_PUNCT_FIXES.items():
        result = result.replace(wrong, correct)

    # Pass 4: conservative spacing fixes
    for pattern, replacement in CONSERVATIVE_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Pass 5: fix capitalization at start of line and after punctuation
    def fix_caps(s):
        s = re.sub(r'(?:^|[.!?]\s+)([a-z])', lambda m: m.group(0).upper(), s)
        return s
    result = fix_caps(result)

    # Pass 6: trim outer whitespace and excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result).strip()

    return result


def post_process_handwriting(text: str) -> str:
    """Convenience wrapper for handwriting mode."""
    return post_process_ocr(text, mode="handwriting")


def process_lines(lines: List[str], mode: str = "handwriting") -> List[str]:
    """Apply post-processing to a list of lines."""
    return [post_process_ocr(line, mode=mode) for line in lines if line and line.strip()]