import hashlib
import re
from pathlib import Path

from modules.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MIN_EXTRACTED_CHARACTERS,
    MIN_EXTRACTED_WORDS,
)

# =========================
# File basics
# =========================

def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().replace(".", "")


def generate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


# =========================
# Upload validation
# =========================

def validate_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return False, "No file was uploaded."

    extension = get_file_extension(uploaded_file.name)

    if extension not in ALLOWED_EXTENSIONS:
        return False, "Unsupported file type. Please upload a PDF, DOCX, or TXT file."

    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        return False, f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."

    return True, "File passed initial validation."


# =========================
# Helpers
# =========================

def _find_matches(patterns, text):
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(match.group(0))
    return list(dict.fromkeys(matches))


# =========================
# FUTURE PLAN DETECTION
# =========================

FUTURE_TERMS = [
    r"\bwill\b",
    r"\bplans?\s+to\b",
    r"\bintends?\s+to\b",
    r"\bexpected\s+to\b",
    r"\bnext\s+year\b",
    r"\bover\s+the\s+next\s+\d+",
    r"\b20(2[6-9]|3\d)\b",
]

SENSITIVE_TERMS = [
    r"\bhiring\b",
    r"\bheadcount\b",
    r"\blayoffs?\b",
    r"\brestructuring\b",
    r"\bexpansion\b",
    r"\brevenue\b",
]

def detect_future_company_plans(text):
    future = _find_matches(FUTURE_TERMS, text)
    sensitive = _find_matches(SENSITIVE_TERMS, text)

    if future and sensitive:
        return True, "Document contains forward-looking company plans.", future[:5]

    return False, "", []


# =========================
# SSN DETECTION
# =========================

SSN_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{3}\s\d{2}\s\d{4}\b",
]

def detect_ssns(text):
    matches = _find_matches(SSN_PATTERNS, text)
    if matches:
        return True, "Document contains a Social Security Number.", matches
    return False, "", []


# =========================
# 🚨 CODE DETECTION (NEW)
# =========================

CODE_PATTERNS = [
    r"\bfunction\s*\(",
    r"\bconsole\.log",
    r"\bvar\s+\w+",
    r"\blet\s+\w+",
    r"\bconst\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+",
    r"\bdef\s+\w+",
    r"\bclass\s+\w+",
    r"\breturn\s+",
    r"\bSELECT\s+.*\bFROM\b",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+\w+",
    r"\bDELETE\s+FROM\b",
    r"<script>",
    r"</script>",
    r"\{\s*\}",
]

def detect_code_content(text):
    matches = _find_matches(CODE_PATTERNS, text)

    if len(matches) >= 3:
        return True, "Document appears to contain programming code.", matches[:5]

    return False, "", []


# =========================
# MAIN VALIDATION
# =========================

def validate_extracted_text(text):
    cleaned = text.strip()
    char_count = len(cleaned)
    word_count = len(cleaned.split())

    if char_count < MIN_EXTRACTED_CHARACTERS or word_count < MIN_EXTRACTED_WORDS:
        return False, "Not enough readable text.", char_count, word_count

    # SSN
    blocked, msg, _ = detect_ssns(cleaned)
    if blocked:
        return False, msg, char_count, word_count

    # CODE
    blocked, msg, _ = detect_code_content(cleaned)
    if blocked:
        return False, msg, char_count, word_count

    # FUTURE
    blocked, msg, _ = detect_future_company_plans(cleaned)
    if blocked:
        return False, msg, char_count, word_count

    return True, "Valid", char_count, word_count


# =========================
# Save file
# =========================

def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    path = uploads_dir / filename
    path.write_bytes(file_bytes)

    return str(path)