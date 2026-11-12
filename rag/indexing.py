"""RAG step 1 — Indexing (RAG From Scratch, Part 2).

load PDFs -> split into chunks -> embed each chunk -> store in a vector DB

Run once (and again whenever the papers change):  python -m rag.indexing
"""

from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag.config import COLLECTION, DB_DIR, EMBED_MODEL, PAPERS_DIR

CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # characters shared between neighbouring chunks


REFERENCE_HEADINGS = {"references", "bibliography"}


def load_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that has text.

    Stops at the "References" heading: bibliography lists are full of
    keywords, so they score as "similar" to questions while containing
    no actual answers.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        lines = (page.extract_text() or "").splitlines()
        cut = next(
            (i for i, line in enumerate(lines) if line.strip().lower() in REFERENCE_HEADINGS),
            None,
        )
        text = " ".join(" ".join(lines[:cut]).split())  # collapse whitespace
        if text:
            pages.append((number, text))
        if cut is not None:
            break  # everything after the references heading is skipped
    return pages


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping windows so ideas cut at a boundary
    still appear whole in at least one chunk."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {PAPERS_DIR}. Run: python -m rag.download_papers")

    ids, documents, metadatas = [], [], []
    for pdf in pdfs:
        for page_number, page_text in load_pages(pdf):
            for i, chunk in enumerate(chunk_text(page_text)):
                ids.append(f"{pdf.stem}-p{page_number}-c{i}")
                documents.append(chunk)
                metadatas.append({"source": pdf.stem, "page": page_number})
        print(f"loaded {pdf.name}")

    print(f"embedding {len(documents)} chunks with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # Start fresh each time so re-running doesn't create duplicates.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    # Remember which model built the index, so retrieval can refuse a mismatch.
    collection = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL}
    )

    batch = 1000  # Chroma limits how much can be added in one call
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i : i + batch],
            documents=documents[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    print(f"stored {collection.count()} chunks in {DB_DIR}")


if __name__ == "__main__":
    main()
