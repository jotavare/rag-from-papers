"""Shared paths and settings, so every script works from any folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"  # downloaded PDFs (not committed)
DB_DIR = ROOT / "data" / "chroma_db"  # vector database (not committed)

COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs on CPU
