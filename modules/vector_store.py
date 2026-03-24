from typing import List, Dict

import chromadb

from modules.config import DATA_DIR

CHROMA_DIR = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "knowledge_base_chunks"


def get_chroma_client():
    """
    Return a persistent Chroma client stored locally in the data folder.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_or_create_collection():
    """
    Get the main chunk collection, creating it if needed.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_chunk_records(
    document_id: str,
    filename: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> Dict[str, List]:
    """
    Build Chroma-ready ids, documents, embeddings, and metadata lists.
    """
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{document_id}_chunk_{i}")
        documents.append(chunk)
        metadatas.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
            }
        )

    return {
        "ids": ids,
        "documents": documents,
        "embeddings": embeddings,
        "metadatas": metadatas,
    }


def add_document_chunks(
    document_id: str,
    filename: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> None:
    """
    Store all chunks for a document in Chroma.
    """
    collection = get_or_create_collection()

    payload = build_chunk_records(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
    )

    collection.add(
        ids=payload["ids"],
        documents=payload["documents"],
        embeddings=payload["embeddings"],
        metadatas=payload["metadatas"],
    )


def get_collection_count() -> int:
    """
    Return the total number of chunk records currently stored.
    """
    collection = get_or_create_collection()
    return collection.count()