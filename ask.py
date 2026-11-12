"""Steps 2 and 3 of RAG: retrieval + generation.

question -> embed -> find similar chunks -> give them to Claude -> answer
"""

import os
import sys

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

from ingest import COLLECTION, DB_DIR, EMBED_MODEL

TOP_K = 5  # how many chunks to give Claude
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You answer questions about AI research papers.
Use ONLY the numbered sources provided. Cite them like [1], [2].
If the sources don't contain the answer, say you don't know."""


def retrieve(question: str) -> list[dict]:
    """Embed the question and return the TOP_K most similar chunks."""
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([question]).tolist()

    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    result = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def generate(question: str, chunks: list[dict]) -> str:
    """Put the retrieved chunks in the prompt and ask Claude."""
    sources = "\n\n".join(
        f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"}],
    )
    return response.content[0].text


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python ask.py "your question"')
    question = " ".join(sys.argv[1:])

    chunks = retrieve(question)
    print("Retrieved sources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['source']} p.{c['page']}  (distance {c['distance']:.3f})")
    print()
    print(generate(question, chunks))


if __name__ == "__main__":
    main()
