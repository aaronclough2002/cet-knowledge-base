from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


# =========================
# PDF Extraction
# =========================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file with safety checks.
    Returns empty string if extraction fails or is too weak.
    """
    text_parts = []

    try:
        pdf_stream = BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception:
                # Skip problematic pages
                continue

    except Exception:
        # Completely failed to read PDF
        return ""

    full_text = "\n".join(text_parts).strip()

    # 🔴 CRITICAL: detect weak extraction (scanned/image PDFs)
    if len(full_text) < 50:
        return ""

    return full_text


# =========================
# DOCX Extraction
# =========================

def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file.
    Returns empty string if extraction fails.
    """
    try:
        text_parts = []

        docx_stream = BytesIO(file_bytes)
        document = Document(docx_stream)

        for paragraph in document.paragraphs:
            if paragraph.text:
                text_parts.append(paragraph.text)

        full_text = "\n".join(text_parts).strip()

        if len(full_text) < 10:
            return ""

        return full_text

    except Exception:
        return ""


# =========================
# TXT Extraction
# =========================

def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from a TXT file.
    """
    try:
        text = file_bytes.decode("utf-8", errors="ignore").strip()

        if len(text) < 5:
            return ""

        return text

    except Exception:
        return ""


# =========================
# ROUTER
# =========================

def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Route extraction based on file extension.
    Returns extracted text or empty string if failure.
    """
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)

    if extension == ".docx":
        return extract_text_from_docx(file_bytes)

    if extension == ".txt":
        return extract_text_from_txt(file_bytes)

    return ""