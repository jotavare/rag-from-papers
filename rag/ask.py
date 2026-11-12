"""Ask a question: (optional planning), retrieval, then generation.

Usage:  python -m rag.ask "your question"
        python -m rag.ask --plan "your question"   # let the LLM plan the search first
"""

import argparse

from rag.generation import BACKEND, generate
from rag.retrieval import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--plan", action="store_true", help="plan the search with the LLM first")
    args = parser.parse_args()
    question = " ".join(args.question)

    if args.plan:
        from rag.planning import plan_search

        plan = plan_search(question)
        print(f"Plan: paper={plan['paper'] or 'all'}  query={plan['query']!r}\n")
        chunks = retrieve(plan["query"], paper=plan["paper"])
    else:
        chunks = retrieve(question)

    print("Retrieved sources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['source']} p.{c['page']}  (distance {c['distance']:.3f})")
    print(f"\nAnswer ({BACKEND}):\n")
    print(generate(question, chunks))


if __name__ == "__main__":
    main()
