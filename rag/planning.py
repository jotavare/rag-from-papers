"""Optional step before retrieval — Planning (the "agent" step).

question -> LLM -> search plan {paper, query} -> retrieval searches only that paper

Plain retrieval can't read the question: "proposed by Lewis et al." means
nothing to an embedding. The LLM can, so it picks which paper to search and
rewrites the question into a better search phrase. If it can't tell, it
returns no paper and retrieval searches everything, as before.
"""

import json
import re

from rag.download_papers import PAPERS
from rag.generation import complete

CATALOG = {name: title for name, title in PAPERS.values()}

PLAN_SYSTEM_PROMPT = """You plan searches over a small library of AI research papers.
Reply with ONLY a JSON object, no other text:
{"paper": "<paper id or null>", "query": "<search phrase>"}

- paper: the one paper the question is about, if it's clear from the
  question (a title, an author, a model name). Otherwise null.
- query: a short search phrase with the key terms likely to appear in
  the passage that answers the question."""


def build_plan_prompt(question: str) -> str:
    papers = "\n".join(f"- {name}: {title}" for name, title in CATALOG.items())
    return f"Papers:\n{papers}\n\nQuestion: {question}"


def parse_plan(reply: str, question: str) -> dict:
    """Read the JSON plan; fall back to a plain search if anything is off."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        plan = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        plan = {}
    paper = plan.get("paper")
    return {
        "paper": paper if paper in CATALOG else None,
        "query": (plan.get("query") or "").strip() or question,
    }


def plan_search(question: str) -> dict:
    return parse_plan(complete(build_plan_prompt(question), PLAN_SYSTEM_PROMPT), question)
