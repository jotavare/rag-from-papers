"""RAG step 2 — Retrieval (RAG From Scratch, Part 3).

question -> embed (same model as indexing) -> find the most similar chunks
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from indexing import COLLECTION, DB_DIR, EMBED_MODEL

TOP_K = 5  # how many chunks to return


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)  # loaded once, reused for every question


@lru_cache(maxsize=1)
def _collection():
    return chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks most similar to the question."""
    query_embedding = _model().encode([question]).tolist()

    collection = _collection()
    result = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
