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


# ---------- STYLING ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.1rem;
        padding-bottom: 1rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .top-note {
        color: #94a3b8;
        margin-bottom: 1rem;
        max-width: 920px;
        line-height: 1.5;
        font-size: 0.98rem;
    }

    .answer-card {
        border: 1px solid #ffffff;
        padding: 14px;
        border-radius: 8px;
        background: #0f172a;
        line-height: 1.55;
    }

    .empty-state {
        border: 1px dashed #ffffff;
        padding: 14px;
        border-radius: 8px;
        color: #94a3b8;
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 8px;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 8px;
    }

    div[data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

# ---------- TOP ----------
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

    display_name = st.text_input(
        "Document name",
        placeholder="Enter clean name",
    )

    upload_clicked = st.button(
        "Add to library",
        disabled=(uploaded_file is None),
    )

    if upload_clicked and uploaded_file is not None:
        is_valid, message = validate_uploaded_file(uploaded_file)

        if not is_valid:
            st.error(message)
        else:
            file_bytes = uploaded_file.getvalue()
            file_hash = generate_file_hash(file_bytes)

            if hash_exists(file_hash):
                st.info("This exact file is already in the shared library.")
            else:
                extracted_text = extract_text(uploaded_file.name, file_bytes)
                valid, msg, char_count, word_count = validate_extracted_text(extracted_text)

                if not valid:
                    st.error(msg)
                else:
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
    st.markdown(
        f'<div class="answer-card">{st.session_state["answer"]}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("sources"):
        st.markdown("### Sources")
        for source in st.session_state["sources"]:
            st.caption(Path(source).stem)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# ---------- LIBRARY ----------
st.subheader("Library")

records = sorted(load_metadata(), key=lambda r: r.get("uploaded_at", ""), reverse=True)

if records:
    controls_col1, controls_col2, controls_col3 = st.columns([5, 1.2, 1.2], gap="small")

    with controls_col1:
        search = st.text_input(
            "Search documents",
            placeholder="Search documents",
            label_visibility="collapsed",
        )

    filtered = [
        record for record in records
        if search.lower() in (
            f"{get_display_name(record)} {record.get('filename', '')} {record.get('summary', '')}".lower()
        )
    ]

    if filtered:
        with controls_col2:
            page_size = st.selectbox("Rows", [5, 10, 20], index=1)

        total_pages = max(1, ceil(len(filtered) / page_size))

        with controls_col3:
            page = st.selectbox("Page", list(range(1, total_pages + 1)))

        start = (page - 1) * page_size
        end = start + page_size
        page_records = filtered[start:end]

        st.caption(f"Showing {start + 1}–{min(end, len(filtered))} of {len(filtered)} documents")

        table_df = build_library_dataframe(page_records)

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=min(420, 56 + 35 * len(table_df)),
        )

        downloadable_records = [
            record for record in page_records
            if Path(record.get("stored_file_path", "")).exists()
        ]

        if downloadable_records:
            dl_col1, dl_col2 = st.columns([5, 1], gap="small")

            with dl_col1:
                selected_download_name = st.selectbox(
                    "Document to download",
                    options=[get_display_name(record) for record in downloadable_records],
                    label_visibility="collapsed",
                )

            selected_record = next(
                record for record in downloadable_records
                if get_display_name(record) == selected_download_name
            )

            with dl_col2:
                with open(selected_record["stored_file_path"], "rb") as file_handle:
                    st.download_button(
                        label="Download",
                        data=file_handle.read(),
                        file_name=selected_record["filename"],
                        help="Download selected document",
                        use_container_width=True,
                        key=f"download_{page}_{selected_record['filename']}",
                    )
    else:
        st.markdown(
            '<div class="empty-state">No documents matched your search.</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="empty-state">No documents yet</div>',
        unsafe_allow_html=True,
    )

# ---------- BOTTOM SPACE ----------
st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)