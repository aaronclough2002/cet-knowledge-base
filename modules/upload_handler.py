import hashlib
from pathlib import Path

from modules.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MIN_EXTRACTED_CHARACTERS,
    MIN_EXTRACTED_WORDS,
)


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without the dot."""
    return Path(filename).suffix.lower().replace(".", "")


def generate_file_hash(file_bytes: bytes) -> str:
    """Generate a SHA-256 hash for the uploaded file."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_uploaded_file(uploaded_file):
    """
    Validate an uploaded file against project rules.

    Returns:
        (is_valid: bool, message: str)
    """
    if uploaded_file is None:
        return False, "No file was uploaded."

    extension = get_file_extension(uploaded_file.name)

    if extension not in ALLOWED_EXTENSIONS:
        return (
            False,
            "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
        )

    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    return True, "File passed initial validation."


def validate_extracted_text(extracted_text: str):
    """
    Validate that extracted text contains enough readable content
    to be useful for downstream processing.

    Returns:
        (is_valid: bool, message: str, char_count: int, word_count: int)
    """
    cleaned_text = extracted_text.strip()
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split())

    if char_count < MIN_EXTRACTED_CHARACTERS or word_count < MIN_EXTRACTED_WORDS:
        return (
            False,
            "This file appears to have too little readable text to process reliably. "
            "Please upload a text-based PDF, DOCX, or TXT file with more readable content.",
            char_count,
            word_count,
        )

    return True, "Extracted text passed content checks.", char_count, word_count

def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """
    Save the original uploaded file to local storage and return its path.
    """
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / filename
    file_path.write_bytes(file_bytes)

    return str(file_path)