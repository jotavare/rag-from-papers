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

## Next steps

- [ ] Contextual Retrieval (Anthropic) — add context to each chunk before embedding
- [ ] Query rewriting / multi-query (RAG From Scratch part 5+)
- [ ] Re-ranking retrieved chunks
- [ ] A small evaluation set to measure whether changes actually help
