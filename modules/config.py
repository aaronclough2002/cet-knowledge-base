from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"

# Upload rules
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_EXTRACTED_CHARACTERS = 200
MIN_EXTRACTED_WORDS = 40
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 50
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# App branding
APP_NAME = "Enterprise Knowledge Base Demo"
APP_VERSION = "v1.0 Demo"
BUILT_BY = "Built by Aaron Clough"