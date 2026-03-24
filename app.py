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

# ---------- HEADER ----------
st.title(APP_NAME)
st.caption(f"{APP_VERSION} Demo | Built by {BUILT_BY}")

col1, col2 = st.columns(2, gap="large")

# ---------- UPLOAD ----------
with col1:
    st.subheader("Upload document")

    # 🔥 NEW: EXPLANATION PANEL
    with st.expander("ℹ️ What gets rejected?"):
        st.markdown("""
        The system enforces validation at upload to prevent sensitive or inappropriate content from entering the knowledge base.

        **Files will be rejected if they contain:**

        - **Forward-looking company plans**
          - Hiring plans, layoffs, restructuring, expansion
          - Workforce changes tied to future dates (e.g., 2026+)
          - Strategy, projections, or planning language

        - **Sensitive personal data**
          - Social Security Numbers (SSNs)

        - **Script or dialogue-style content**
          - Character dialogue (e.g., "John:")
          - Scene formats (INT., EXT., FADE IN, CUT TO)

        - **Unreadable or low-quality files**
          - Scanned PDFs with no extractable text
          - Files with insufficient readable content

        **Allowed content includes:**
        - Historical reports
        - Completed initiatives
        - Past HR or financial summaries
        """)

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

        # STEP 1 — File validation
        is_valid, message = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(message)
            st.stop()

        file_bytes = uploaded_file.getvalue()
        file_hash = generate_file_hash(file_bytes)

        # STEP 2 — Duplicate check
        if hash_exists(file_hash):
            st.info("This exact file is already in the shared library.")
            st.stop()

        # STEP 3 — Extract text
        extracted_text = extract_text(uploaded_file.name, file_bytes)

        # 🔴 Extraction failure check
        if not extracted_text or not extracted_text.strip():
            st.error(
                "Upload rejected: Could not extract readable text from this file. "
                "This typically occurs with scanned PDFs or unreadable documents."
            )
            st.stop()

        # STEP 4 — Content validation
        valid, msg, char_count, word_count = validate_extracted_text(extracted_text)

        if not valid:
            st.error(f"Upload rejected: {msg}")
            st.stop()

        # STEP 5 — Process document
        clean_display_name = (
            Path(display_name).stem if display_name.strip()
            else Path(uploaded_file.name).stem
        )

        chunks = chunk_text(
            extracted_text,
            CHUNK_SIZE_WORDS,
            CHUNK_OVERLAP_WORDS,
        )

        with st.spinner("Processing document..."):
            try:
                summary = generate_document_summary(extracted_text)
            except Exception:
                summary = "Summary unavailable."

            embeddings = embed_chunks(chunks)
            doc_id = generate_document_id()
            stored_path = save_uploaded_file(file_bytes, uploaded_file.name)

            add_document_chunks(
                doc_id,
                uploaded_file.name,
                chunks,
                embeddings,
            )

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

records = sorted(
    load_metadata(),
    key=lambda r: r.get("uploaded_at", ""),
    reverse=True,
)

if records:
    df = pd.DataFrame([
        {
            "Name": r.get("display_name") or Path(r.get("filename")).stem,
            "Summary": (r.get("summary") or "")[:100],
            "Uploaded": r.get("uploaded_at"),
        }
        for r in records
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No documents yet.")