"""Shared paths and settings, so every script works from any folder."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"  # downloaded PDFs (not committed)
DB_DIR = ROOT / "data" / "chroma_db"  # vector database (not committed)

COLLECTION = "papers"
# Embedding model used for indexing AND retrieval; change it, then re-index.
#   all-MiniLM-L6-v2        384 dims, reads up to 256 tokens (original baseline)
#   BAAI/bge-small-en-v1.5  384 dims, reads up to 512 tokens
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# Some models expect an instruction in front of search queries (not chunks).
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
QUERY_PREFIX = QUERY_PREFIXES.get(EMBED_MODEL, "")
