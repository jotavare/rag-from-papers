"""RAG step 2 — Retrieval (RAG From Scratch, Part 3).

question -> embed (same model as indexing) -> find the most similar chunks
"""

import chromadb
from sentence_transformers import SentenceTransformer

from indexing import COLLECTION, DB_DIR, EMBED_MODEL

TOP_K = 5  # how many chunks to return


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k chunks most similar to the question."""
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([question]).tolist()

    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    result = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
