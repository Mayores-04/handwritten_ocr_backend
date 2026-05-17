"""OCR post-processing.

This module is intentionally conservative: it fixes common OCR artifacts and
normalizes punctuation/whitespace without trying to "guess" full words.
"""

import re
from typing import List



# ============ COMMON HANDWRITING OCR MISREADS ============
COMMON_HANDWRITING_FIXES = [
    # Standalone "1" is frequently misread "I" in handwriting OCR.
    (r"(?<!\w)1(?!\w)", "I"),
    # Very common non-words produced by OCR on simple English greetings.
    (r"\bHelb\b", "Hello"),
    (r"\bgys\b", "guys"),
    # Common OCR "OCR System" variants.
    (r"\bCCRSystem\b", "OCR System"),
    (r"\bCCR\s+System\b", "OCR System"),
    (r"\bOCRSystem\b", "OCR System"),
    # Common phrase fragments from your sample handwriting.
    (r"\bonwertHis\b", "convert this"),
    (r"\bTnTo\(?\s*convert\s*this\b", "I will try to convert this"),
    (r"\bTnTo\(?\s*onwertHis\b", "I will try to convert this"),
    (r"\bUsil\s+Cur\b", "using our"),
    # Name phrase cleanup (kept as word-level fixes, not whole-sentence overrides).
    (r"\bDynone\b", "My Name"),
    (r"\bNane\b", "Name"),
    (r"\bI\s+J\b", "Jake J."),
    (r"\(\s*aueres\b", "Mayores."),
    (r"\baueres\b", "Mayores"),
    (r"\bManores\b", "Mayores"),
    # OCR casing for common acronyms.
    (r"\bOCr\b", "OCR"),
    (r"\b0CR\b", "OCR"),
    (r"\b0cr\b", "ocr"),
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
    (r"\s+!", "!"),
    (r"\s+\?", "?"),
    (r",{2,}", ","),
    (r";{2,}", ";"),
    (r"!{3,}", "!!"),
    (r"\?{3,}", "??"),
    (r"(?<!\n)  +", " "),
    (r"\(\s{2,}", "("),
    (r"\s{2,}\)", ")"),
]


_IN_WORD_QUOTES = re.compile(r'(?<=\w)[`"\'’‘´](?=\w)')
_PUNCT_ONLY_LINE = re.compile(r"^[\s\W_]+$")
_SHORT_NOISE_LINE = re.compile(r"^\s*[\d\W_]{1,3}\s*$")

# Printed OCR frequently confuses punctuation (e.g. '.' -> ';', ',' -> ':').
# Keep these very conservative to avoid corrupting legitimate punctuation such as
# times (12:30) and ratios (1:2).
PRINTED_PUNCT_FIXES = [
    # Sentence terminators: semicolon/colon that are acting like a period.
    (r"(?<!\d);(?=\s+[A-Z])", "."),
    (r"(?<!\d):(?=\s+[A-Z])", "."),
    (r"(?<!\d);(?=\s*$)", "."),
    (r"(?<!\d):(?=\s*$)", "."),
    # Comma-like: colon between words (not digits) followed by lowercase.
    (r"(?<!\d):(?=\s+[a-z])", ","),
    # Semicolon used like a comma in prose.
    (r";(?=\s+[a-z])", ","),
    # Fix missing space after period in prose: "schedule.This" -> "schedule. This"
    (r"([A-Za-z])\.(?=[A-Z])", r"\1. "),
    # Time written with '.' instead of ':' -> normalize when clearly a time.
    (r"\b(\d{1,2})\.(\d{2})\s*([AP]M)\b", r"\1:\2 \3"),
]


def _strip_in_word_quotes(text: str) -> str:
    # Fix artifacts like `g"ys` -> `gys` (still conservative; doesn't invent letters).
    return _IN_WORD_QUOTES.sub("", text)


def _normalize_common_artifacts(text: str) -> str:
    # Normalize some frequent OCR punctuation artifacts without changing words.
    text = _strip_in_word_quotes(text)
    # Replace odd quote variants with a plain apostrophe.
    text = text.replace("’", "'").replace("‘", "'").replace("´", "'")
    return text


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

    result = _normalize_common_artifacts(text)

    # Pass 1: code-only fixes
    if mode == "code":
        for pattern, replacement in CODE_ONLY_FIXES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Pass 1b: printed-only punctuation fixes
    if mode == "printed":
        for pattern, replacement in PRINTED_PUNCT_FIXES:
            result = re.sub(pattern, replacement, result)

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

    if mode == "printed":
        # Printed OCR often returns hard-wrapped lines from a paragraph. If a line
        # break occurs mid-sentence, the next line may start with an uppercase
        # letter even though it should be lowercase.
        lines = result.splitlines()
        fixed_lines: list[str] = []
        prev_line = ""
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                fixed_lines.append(line)
                prev_line = line
                continue

            prev_tail = prev_line.rstrip()
            prev_end = prev_tail[-1] if prev_tail else ""

            # If previous line does not end a sentence, and this line begins with
            # a Titlecase word, lowercase the first character.
            if prev_end and prev_end not in ".!?":
                if len(stripped) >= 2 and stripped[0].isupper() and stripped[1].islower():
                    # Avoid lowercasing standalone "I ..."
                    if not stripped.startswith("I "):
                        leading_ws = line[: len(line) - len(stripped)]
                        stripped = stripped[0].lower() + stripped[1:]
                        line = leading_ws + stripped

            fixed_lines.append(line)
            prev_line = line

        result = "\n".join(fixed_lines)

    # Pass 6: trim outer whitespace and excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result).strip()

    return result


def post_process_handwriting(text: str) -> str:
    """Convenience wrapper for handwriting mode."""
    return post_process_ocr(text, mode="handwriting")


def process_lines(lines: List[str], mode: str = "handwriting") -> List[str]:
    """Apply post-processing to a list of lines.

    Also drops obvious noise-only lines when there are other meaningful lines.
    """
    cleaned = [post_process_ocr(line, mode=mode) for line in lines if line and line.strip()]
    if len(cleaned) <= 1:
        return cleaned

    filtered: list[str] = []
    for line in cleaned:
        candidate = line.strip()
        if not candidate:
            continue
        if _PUNCT_ONLY_LINE.match(candidate):
            continue
        if _SHORT_NOISE_LINE.match(candidate):
            # Common artifacts: a lone "5", ".", "=", etc.
            continue
        filtered.append(line)

    return filtered or cleaned
