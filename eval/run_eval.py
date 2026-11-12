"""Evaluate the RAG pipeline on a fixed set of questions.

Retrieval (always, free and fast):
  hit@5  did the expected paper appear in the top 5 chunks?
  MRR    mean reciprocal rank: 1.0 if the first chunk is from the right paper,
         0.5 if the first correct one is 2nd, 0.33 if 3rd... (0 if missing)

Answers (only with --answers, uses LLM_BACKEND):
  pass   the answer contains every required keyword group (any alternative
         in a group counts). Crude, but cheap and repeatable.

Planning (only with --plan, uses LLM_BACKEND):
  the LLM picks a paper and a search phrase before retrieval

Run from the project root:
  python -m eval.run_eval
  python -m eval.run_eval --answers
  python -m eval.run_eval --plan --answers
"""

import argparse
import json
from pathlib import Path

from rag.retrieval import TOP_K, retrieve

QUESTIONS_FILE = Path(__file__).with_name("questions.json")


def keyword_pass(answer: str, must_include: list[list[str]]) -> bool:
    text = answer.lower().replace("’", "'")
    return all(any(alt.lower() in text for alt in group) for group in must_include)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also generate and check answers")
    parser.add_argument("--plan", action="store_true", help="plan each search with the LLM first")
    args = parser.parse_args()

    if args.answers or args.plan:
        from rag.generation import BACKEND, generate
    if args.plan:
        from rag.planning import plan_search

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    retrieval_rows = [q for q in questions if q["source"]]

    hits, rr_total, answer_passes = 0, 0.0, 0
    for n, q in enumerate(questions, start=1):
        line = f"{n:>2}. "
        if args.plan:
            plan = plan_search(q["question"])
            chunks = retrieve(plan["query"], TOP_K, paper=plan["paper"])
            line += f"plan {plan['paper'] or 'all':<26} "
        else:
            chunks = retrieve(q["question"], TOP_K)
        sources = [c["source"] for c in chunks]

        if q["source"]:
            rank = sources.index(q["source"]) + 1 if q["source"] in sources else None
            hits += rank is not None
            rr_total += 1 / rank if rank else 0
            line += f"rank {rank or '-':<2} "
        else:
            line += "(no source) "

        if args.answers:
            answer = generate(q["question"], chunks)
            ok = keyword_pass(answer, q["must_include"])
            answer_passes += ok
            line += "PASS " if ok else "FAIL "

        print(line + q["question"])
        if args.answers and not ok:
            print("      answer: " + " ".join(answer.split())[:200])

    total = len(retrieval_rows)
    print()
    print(f"Retrieval  hit@{TOP_K}: {hits}/{total} ({hits / total:.0%})   MRR: {rr_total / total:.2f}")
    if args.plan:
        print(f"Planning   ({BACKEND}): on")
    if args.answers:
        print(f"Answers    ({BACKEND}): {answer_passes}/{len(questions)} passed keyword check")


if __name__ == "__main__":
    main()
