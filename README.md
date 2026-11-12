# rag-from-papers

A minimal Retrieval-Augmented Generation (RAG) app that answers questions
about classic AI research papers — with a local model (free, offline) or Claude.

Built while following LangChain's *RAG From Scratch* series (parts 1–4).

## How it works

Each file is one step of the pipeline:

| File | Step | What happens |
|------|------|--------------|
| `indexing.py` | Indexing (Part 2) | PDFs → text (reference sections skipped) → overlapping 1000-char chunks → embeddings (`all-MiniLM-L6-v2`) → local Chroma DB |
| `retrieval.py` | Retrieval (Part 3) | The question is embedded the same way; the 5 most similar chunks are fetched |
| `generation.py` | Generation (Part 4) | The chunks go into the prompt as numbered sources; an LLM answers and cites them |
| `ask.py` | — | Runs retrieval, then generation |
| `download_papers.py` | Setup | Fetches the papers from arXiv |

## Papers

Downloaded from arXiv by `download_papers.py` (not committed to the repo):
Attention Is All You Need, BERT, Sentence-BERT, RAG (Lewis et al. 2020),
Chain-of-Thought, ReAct, and a RAG survey (Gao et al. 2023).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python download_papers.py
python indexing.py
```

## Choosing the model that writes answers

Set `LLM_BACKEND` (retrieval is the same for all of them):

| Backend | Needs | Notes |
|---------|-------|-------|
| `ollama` (default) | [Ollama](https://ollama.com) + `ollama pull llama3.2:3b` | Free and offline; ~1 min per answer on CPU. Change model with `OLLAMA_MODEL` |
| `claude-code` | [Claude Code](https://claude.com/claude-code) signed in to a Claude subscription | Uses your plan's usage limits; no API key |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic API; change model with `CLAUDE_MODEL` |

```bash
python ask.py "What problem does retrieval-augmented generation solve?"

# PowerShell: pick another backend for this terminal
$env:LLM_BACKEND = "claude-code"
python ask.py "How does ReAct combine reasoning and acting?"
```

## Evaluation

`eval/questions.json` holds 14 questions with known answers: 13 answerable from a specific paper, plus one off-topic question the app should refuse.

```bash
python eval/run_eval.py             # retrieval only: free, a few seconds
python eval/run_eval.py --answers   # also generate answers with LLM_BACKEND and keyword-check them
```

Baseline (fixed-size chunks, all-MiniLM-L6-v2, top 5, Claude via `claude-code`):

| Metric | Score |
|--------|-------|
| Retrieval hit@5 (right paper in top 5) | 13/13 |
| Retrieval MRR | 0.89 |
| Answers passing keyword check | 12/14 |

Both failures ask about the original RAG paper (Lewis et al.). The paper was retrieved, but not the passage holding the answer, and the model correctly replied that the sources don't say. Finding the right paper isn't enough; the right chunk has to be found too.

## Explainer

[`rag-explainer.html`](rag-explainer.html) is an illustrated walkthrough of the three stages with this project's real numbers. Open it in a browser.

## Next steps

- [ ] Contextual Retrieval (Anthropic): add context to each chunk before embedding, then re-run the eval
- [ ] Re-ranking retrieved chunks
- [ ] Query rewriting / multi-query (RAG From Scratch part 5+)
- [x] A small evaluation set to measure whether changes actually help
