from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    text_parts = []

    pdf_stream = BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    text_parts = []

    docx_stream = BytesIO(file_bytes)
    document = Document(docx_stream)

    for paragraph in document.paragraphs:
        if paragraph.text:
            text_parts.append(paragraph.text)

    return "\n".join(text_parts).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a TXT file."""
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Route extraction based on file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == ".docx":
        return extract_text_from_docx(file_bytes)
    if extension == ".txt":
        return extract_text_from_txt(file_bytes)

    raise ValueError(f"Unsupported file type for extraction: {extension}")