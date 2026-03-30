from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.config import (
    APP_NAME,
    APP_VERSION,
    BUILT_BY,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)
from modules.upload_handler import (
    validate_uploaded_file,
    validate_extracted_text,
    generate_file_hash,
    save_uploaded_file,
)
from modules.extractors import extract_text
from modules.metadata import (
    hash_exists,
    add_document_record,
    generate_document_id,
    load_metadata,
)
from modules.chunker import chunk_text
from modules.embeddings import embed_chunks
from modules.vector_store import add_document_chunks
from modules.qa_engine import (
    retrieve_relevant_chunks,
    generate_grounded_answer,
    get_source_documents,
    generate_document_summary,
)

st.set_page_config(page_title=APP_NAME, page_icon="📚", layout="wide")


# ---------- GREEN THEME ----------
st.markdown("""
<style>
:root {
    --bg: linear-gradient(180deg, #eef5ef 0%, #e7f0e9 100%);
    --panel: rgba(255, 255, 255, 0.94);
    --panel-soft: #f5faf6;
    --text: #1c2f25;
    --muted: #5f7165;
    --accent: #4a765b;
    --accent-dark: #335340;
    --line: rgba(60, 90, 69, 0.12);
    --shadow: 0 14px 34px rgba(0, 0, 0, 0.06);
    --radius: 22px;
}

/* Background */
.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Layout */
.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Typography */
h1, h2, h3, label, p, div, span {
    color: var(--text) !important;
}

/* Cards */
.section-card {
    background: var(--panel);
    border-radius: var(--radius);
    padding: 22px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,0.4);
    margin-bottom: 18px;
}

/* Top description */
.top-note {
    color: #24362b;
    margin-bottom: 1rem;
    line-height: 1.6;
    font-size: 1rem;
}

/* Buttons */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background: var(--accent-dark) !important;
}

/* Inputs */
textarea, input {
    border-radius: 12px !important;
    border: 1px solid #d8e5db !important;
}

/* Answer */
.answer-card {
    background: var(--panel-soft);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px;
    line-height: 1.6;
}

/* Empty */
.empty-state {
    background: var(--panel-soft);
    border: 1px dashed #cfded2;
    padding: 18px;
    border-radius: 16px;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)


# ---------- HELPERS ----------
def format_uploaded_date(uploaded_at: str) -> str:
    if not uploaded_at:
        return "Unknown"
    try:
        return datetime.fromisoformat(uploaded_at).strftime("%m/%d/%y %I:%M %p")
    except Exception:
        return uploaded_at


def get_display_name(record: dict) -> str:
    display_name = record.get("display_name", "").strip()
    if display_name:
        return display_name
    return Path(record.get("filename", "Untitled")).stem


def shorten_summary(summary: str, max_words: int = 15) -> str:
    if not summary:
        return "Summary unavailable."
    words = summary.split()
    return " ".join(words[:max_words]) + "..." if len(words) > max_words else summary


def normalize_display_name(name: str, filename: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        cleaned = Path(filename).stem
    return Path(cleaned).stem.strip()


def get_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.replace(".", "").upper()
    return suffix if suffix else "FILE"


def build_library_dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": get_display_name(record),
                "Summary": shorten_summary(record.get("summary", "")),
                "Uploaded": format_uploaded_date(record.get("uploaded_at", "")),
                "Type": get_file_type(record.get("filename", "")),
            }
            for record in records
        ]
    )


# ---------- HEADER ----------
st.title(APP_NAME)
st.caption(f"{APP_VERSION} Demo | Built by {BUILT_BY}")

st.markdown(
    """
    <div class="top-note">
    Build a shared enterprise document library across teams and subsidiaries, and ask questions across all uploaded documents.
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

# ---------- UPLOAD ----------
with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload document")

    uploaded_file = st.file_uploader("Upload", type=["pdf", "docx", "txt"])
    display_name = st.text_input("Document name")

    if st.button("Add to library"):
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_hash = generate_file_hash(file_bytes)

            if hash_exists(file_hash):
                st.info("File already exists.")
                st.stop()

            text = extract_text(uploaded_file.name, file_bytes)
            chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

            embeddings = embed_chunks(chunks)
            doc_id = generate_document_id()

            add_document_chunks(doc_id, uploaded_file.name, chunks, embeddings)

            add_document_record({
                "document_id": doc_id,
                "filename": uploaded_file.name,
                "display_name": normalize_display_name(display_name, uploaded_file.name),
                "uploaded_at": datetime.utcnow().isoformat(),
                "summary": generate_document_summary(text),
            })

            st.success("Added!")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ASK ----------
with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Ask the knowledge base")

    q = st.text_input("Ask")

    if st.button("Ask"):
        matches = retrieve_relevant_chunks(q)
        answer = generate_grounded_answer(q, matches)
        sources = get_source_documents(matches)

        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        for s in sources:
            st.caption(Path(s).stem)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- LIBRARY ----------
st.subheader("Library")

records = load_metadata()

if records:
    df = build_library_dataframe(records)
    st.dataframe(df, use_container_width=True)
else:
    st.markdown('<div class="empty-state">No documents yet</div>', unsafe_allow_html=True)