import json
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from modules.config import DATA_DIR

METADATA_FILE = DATA_DIR / "document_metadata.json"


def ensure_metadata_store_exists() -> None:
    """Create the metadata JSON file if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_FILE.exists():
        METADATA_FILE.write_text("[]", encoding="utf-8")


def load_metadata() -> List[Dict]:
    """Load all document metadata records."""
    ensure_metadata_store_exists()

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(records: List[Dict]) -> None:
    """Save all document metadata records."""
    ensure_metadata_store_exists()

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def generate_document_id() -> str:
    """Generate a unique internal document ID."""
    return str(uuid4())


def add_document_record(record: Dict) -> None:
    """Append a new document record to the metadata store."""
    records = load_metadata()
    records.append(record)
    save_metadata(records)


def hash_exists(file_hash: str) -> bool:
    """Return True if the given file hash already exists."""
    records = load_metadata()
    return any(record.get("file_hash") == file_hash for record in records)