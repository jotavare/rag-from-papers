"""Ask a question: runs retrieval, then generation.

Usage:  python -m rag.ask "your question"
"""

import sys

from rag.generation import BACKEND, generate
from rag.retrieval import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m rag.ask "your question"')
    question = " ".join(sys.argv[1:])

    chunks = retrieve(question)
    print("Retrieved sources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['source']} p.{c['page']}  (distance {c['distance']:.3f})")
    print(f"\nAnswer ({BACKEND}):\n")
    print(generate(question, chunks))


if __name__ == "__main__":
    main()
