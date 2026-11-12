"""RAG step 2 — Retrieval (RAG From Scratch, Part 3).

question -> embed (same model as indexing) -> find the most similar chunks
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import COLLECTION, DB_DIR, EMBED_MODEL, QUERY_PREFIX

TOP_K = 5  # how many chunks to return


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)  # loaded once, reused for every question


@lru_cache(maxsize=1)
def _collection():
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    indexed_with = (collection.metadata or {}).get("embed_model")
    if indexed_with and indexed_with != EMBED_MODEL:
        raise SystemExit(
            f"The index was built with {indexed_with}, but EMBED_MODEL is {EMBED_MODEL}. "
            "Vectors from different models can't be compared: run python -m rag.indexing"
        )
    return collection


def retrieve(question: str, top_k: int = TOP_K, paper: str | None = None) -> list[dict]:
    """Return the top_k chunks most similar to the question.

    paper: search only this paper's chunks (set by the planning step).
    """
    query_embedding = _model().encode([QUERY_PREFIX + question]).tolist()

    collection = _collection()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"source": paper} if paper else None,
    )

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
