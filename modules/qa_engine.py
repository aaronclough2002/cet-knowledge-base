import os
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from modules.embeddings import embed_text
from modules.vector_store import get_or_create_collection

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_query_terms(query: str) -> List[str]:
    """
    Extract meaningful query terms for lightweight reranking.
    """
    stop_words = {
        "the", "and", "for", "with", "this", "that", "what", "which", "from",
        "into", "are", "is", "was", "were", "how", "does", "used", "use",
        "project", "system", "document", "about", "their", "there", "main"
    }

    return [
        term.lower()
        for term in re.findall(r"\b\w+\b", query)
        if len(term) > 2 and term.lower() not in stop_words
    ]


def keyword_overlap_score(query: str, text: str) -> int:
    """
    Count how many meaningful query terms appear in the text.
    """
    query_terms = set(extract_query_terms(query))
    text_terms = {
        term.lower()
        for term in re.findall(r"\b\w+\b", text)
        if len(term) > 2
    }
    return len(query_terms.intersection(text_terms))


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve the most relevant chunks from Chroma for a user query.
    Applies lightweight keyword reranking after embedding search.
    """
    if not query.strip():
        return []

    query_embedding = embed_text(query)

    collection = get_or_create_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for doc, metadata, distance in zip(documents, metadatas, distances):
        overlap = keyword_overlap_score(query, doc)
        retrieved.append(
            {
                "chunk_text": doc,
                "metadata": metadata,
                "distance": distance,
                "keyword_overlap": overlap,
            }
        )

    retrieved.sort(
        key=lambda item: (-item.get("keyword_overlap", 0), item.get("distance", 999))
    )

    return retrieved[:top_k]


def build_context_from_matches(matches: List[Dict]) -> str:
    """
    Build a structured context block from retrieved chunk matches.
    """
    context_sections = []

    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        chunk_index = metadata.get("chunk_index", "Unknown")
        chunk_text = match.get("chunk_text", "")

        section = (
            f"[Source {i}]\n"
            f"Filename: {filename}\n"
            f"Chunk Index: {chunk_index}\n"
            f"Content:\n{chunk_text}"
        )
        context_sections.append(section)

    return "\n\n".join(context_sections)


def generate_grounded_answer(query: str, matches: List[Dict]) -> str:
    """
    Generate an answer using only the retrieved document context.
    """
    if not matches:
        return (
            "I don't see enough information in the current document library "
            "to answer that confidently."
        )

    context = build_context_from_matches(matches)

    system_prompt = """
You are answering questions for the Cutting Edge Technologies Knowledge Base demo.

Rules:
- Answer ONLY from the provided document context.
- Do NOT use outside knowledge.
- Do NOT guess or fill gaps.
- If the context is insufficient, say:
  "I don't see enough information in the current document library to answer that confidently."
- Be clear, professional, and concise.
- When helpful, mention the source document names naturally.
"""

    user_prompt = f"""
Question:
{query}

Document Context:
{context}

Provide a grounded answer based only on the document context above.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )

    return response.choices[0].message.content.strip()


def get_source_documents(matches: List[Dict], max_docs: int = 3) -> List[str]:
    """
    Return a compact de-duplicated list of source document names.
    """
    seen = []
    for match in matches:
        metadata = match.get("metadata", {})
        filename = metadata.get("filename")
        if filename and filename not in seen:
            seen.append(filename)

    return seen[:max_docs]
def generate_document_summary(text: str) -> str:
    """
    Generate a concise 1-line summary of a document.
    """
    from openai import OpenAI
    client = OpenAI()

    preview = text[:3000]  # keep it small for speed

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Summarize the document in ONE short sentence (max 15 words). Be clear and specific."
            },
            {
                "role": "user",
                "content": preview
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()