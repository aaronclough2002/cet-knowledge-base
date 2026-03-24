from datetime import datetime
from math import ceil
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
    return Path(record.get("filename", "Untitled document")).stem


def shorten_summary(summary: str, max_words: int = 15) -> str:
    if not summary:
        return "Summary unavailable."
    words = summary.split()
    if len(words) <= max_words:
        return summary
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


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


# ---------- UI ----------
st.title(APP_NAME)
st.caption(f"{APP_VERSION} Demo | Built by {BUILT_BY}")

col1, col2 = st.columns(2, gap="large")

# ---------- UPLOAD ----------
with col1:
    st.subheader("Upload document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

    display_name = st.text_input("Document name", placeholder="Enter clean name")

    upload_clicked = st.button(
        "Add to library",
        disabled=(uploaded_file is None),
    )

    if upload_clicked and uploaded_file is not None:

        # STEP 1: Basic file validation
        is_valid, message = validate_uploaded_file(uploaded_file)

        if not is_valid:
            st.error(message)
            st.stop()

        file_bytes = uploaded_file.getvalue()
        file_hash = generate_file_hash(file_bytes)

        # STEP 2: Duplicate check
        if hash_exists(file_hash):
            st.info("This exact file is already in the shared library.")
            st.stop()

        # STEP 3: Extract text
        extracted_text = extract_text(uploaded_file.name, file_bytes)

        # 🔴 CRITICAL: Extraction safety check
        if not extracted_text or not extracted_text.strip():
            st.error(
                "Could not extract readable text from this file. "
                "Please upload a valid PDF, DOCX, or TXT file."
            )
            st.stop()

        # STEP 4: Content validation (future plans, SSN, scripts)
        valid, msg, char_count, word_count = validate_extracted_text(extracted_text)

        if not valid:
            st.error(msg)
            st.caption(
                "Uploads are blocked if they contain future company plans, "
                "SSNs, or script-style content."
            )
            st.stop()

        # STEP 5: Process document
        clean_display_name = normalize_display_name(display_name, uploaded_file.name)
        chunks = chunk_text(extracted_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

        with st.spinner("Processing document..."):
            try:
                summary = generate_document_summary(extracted_text)
            except Exception:
                summary = "Summary unavailable."

            embeddings = embed_chunks(chunks)
            doc_id = generate_document_id()
            stored_path = save_uploaded_file(file_bytes, uploaded_file.name)

            add_document_chunks(doc_id, uploaded_file.name, chunks, embeddings)

            add_document_record(
                {
                    "document_id": doc_id,
                    "filename": uploaded_file.name,
                    "display_name": clean_display_name,
                    "file_hash": file_hash,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "char_count": char_count,
                    "word_count": word_count,
                    "chunk_count": len(chunks),
                    "embedding_count": len(embeddings),
                    "stored_file_path": stored_path,
                    "summary": summary,
                }
            )

        st.success(f"Added: {clean_display_name}")


# ---------- ASK ----------
with col2:
    st.subheader("Ask the knowledge base")

    with st.form("qa_form", clear_on_submit=False):
        q = st.text_input(
            "Question",
            placeholder="Ask something...",
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("Ask")

    if ask_clicked and q.strip():
        with st.spinner("Thinking..."):
            matches = retrieve_relevant_chunks(q, top_k=5)
            answer = generate_grounded_answer(q, matches)
            sources = get_source_documents(matches)

        st.session_state["answer"] = answer
        st.session_state["sources"] = sources


# ---------- ANSWER ----------
if "answer" in st.session_state:
    st.markdown("## Answer")
    st.markdown(st.session_state["answer"])

    if st.session_state.get("sources"):
        st.markdown("### Sources")
        for source in st.session_state["sources"]:
            st.caption(Path(source).stem)


# ---------- LIBRARY ----------
st.subheader("Library")

records = sorted(load_metadata(), key=lambda r: r.get("uploaded_at", ""), reverse=True)

if records:
    df = build_library_dataframe(records)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No documents yet.")