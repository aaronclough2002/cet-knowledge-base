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

    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."

    return True, "File passed initial validation."


# =========================
# Pattern helpers
# =========================

def _find_matches(patterns, text: str):
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(match.group(0))
    return list(dict.fromkeys(matches))


# =========================
# Future plans detection
# =========================

FUTURE_TERMS = [
    r"\bwill\b",
    r"\bplans?\s+to\b",
    r"\bplanning\s+to\b",
    r"\bintends?\s+to\b",
    r"\bexpected\s+to\b",
    r"\banticipates?\b",
    r"\bforecast(?:ed|s|ing)?\b",
    r"\bproject(?:ed|ion|ions)?\b",
    r"\broadmap\b",
    r"\bstrategy\b",
    r"\bnext\s+year\b",
    r"\bfuture\b",
    r"\bover\s+the\s+next\s+\d+\s+(month|months|year|years)\b",
    r"\bby\s+20\d{2}\b",
    r"\bq[1-4]\s+20\d{2}\b",
    r"\bfy\s*20\d{2}\b",
    r"\b20(2[6-9]|3\d)\b",
]

SENSITIVE_PLAN_TERMS = [
    r"\bhiring\b",
    r"\bheadcount\b",
    r"\bworkforce\b",
    r"\blayoffs?\b",
    r"\breduction\s+in\s+force\b",
    r"\brestructuring\b",
    r"\bexpansion\b",
    r"\bacquisition(s)?\b",
    r"\bbudget(s)?\b",
    r"\brevenue\b",
    r"\bprofit\b",
]

def detect_future_company_plans(text: str):
    normalized_text = " ".join(text.split())

    future_matches = _find_matches(FUTURE_TERMS, normalized_text)
    sensitive_matches = _find_matches(SENSITIVE_PLAN_TERMS, normalized_text)

    if future_matches and sensitive_matches:
        return True, "Document contains forward-looking company plans.", future_matches[:5]

    return False, "", []


# =========================
# SSN detection
# =========================

SSN_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{3}\s\d{2}\s\d{4}\b",
]

def detect_ssns(text: str):
    matches = _find_matches(SSN_PATTERNS, text)

    if matches:
        return True, "Document contains a Social Security Number.", matches

    return False, "", []


# =========================
# Script detection (IMPROVED)
# =========================

SCRIPT_KEYWORDS = [
    r"\bint\.\b",
    r"\bext\.\b",
    r"\bfade in\b",
    r"\bcut to\b",
    r"\bvoiceover\b",
    r"\bscript\b",
]

def detect_script_content(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lowered = text.lower()

    keyword_matches = _find_matches(SCRIPT_KEYWORDS, lowered)

    scene_headings = 0
    uppercase_lines = 0
    parentheticals = 0
    dialogue_colon_lines = 0

    for line in lines[:400]:

        if line.startswith(("INT.", "EXT.", "Int.", "Ext.")):
            scene_headings += 1

        if line.isupper() and len(line.split()) <= 5:
            uppercase_lines += 1

        if line.startswith("(") and line.endswith(")"):
            parentheticals += 1

        # 🔴 NEW: detect "Name:" dialogue
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts[0].split()) <= 3 and len(parts[0]) <= 20:
                dialogue_colon_lines += 1

    score = 0

    if keyword_matches:
        score += 2
    if scene_headings >= 2:
        score += 2
    if uppercase_lines >= 8:
        score += 2
    if parentheticals >= 4:
        score += 1
    if dialogue_colon_lines >= 2:
        score += 2

    if score >= 3:
        return True, "Document appears to be a script or dialogue-based format.", keyword_matches

    return False, "", []


# =========================
# Main validation
# =========================

def validate_extracted_text(text: str):
    cleaned = text.strip()
    char_count = len(cleaned)
    word_count = len(cleaned.split())

    if char_count < MIN_EXTRACTED_CHARACTERS or word_count < MIN_EXTRACTED_WORDS:
        return False, "Not enough readable text.", char_count, word_count

    # SSN
    blocked, msg, matches = detect_ssns(cleaned)
    if blocked:
        return False, msg, char_count, word_count

    # Script
    blocked, msg, matches = detect_script_content(cleaned)
    if blocked:
        return False, msg, char_count, word_count

    # Future plans
    blocked, msg, matches = detect_future_company_plans(cleaned)
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