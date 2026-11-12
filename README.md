# rag-from-papers

A minimal Retrieval-Augmented Generation (RAG) app that answers questions
about classic AI research papers, using Claude to write the answers.

Built while following LangChain's *RAG From Scratch* series (parts 1–4).

## How it works

| Step | File | What happens |
|------|------|--------------|
| Indexing | `ingest.py` | PDFs → text → overlapping 1000-char chunks → embeddings (`all-MiniLM-L6-v2`) → stored in a local Chroma DB |
| Retrieval | `ask.py` | The question is embedded the same way; the 5 most similar chunks are fetched |
| Generation | `ask.py` | The chunks go into the prompt as numbered sources; Claude answers and cites them |

## Papers

Downloaded from arXiv by `download_papers.py` (not committed to the repo):
Attention Is All You Need, BERT, Sentence-BERT, RAG (Lewis et al. 2020),
Chain-of-Thought, ReAct, and a RAG survey (Gao et al. 2023).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Set your Anthropic API key (get one at https://console.anthropic.com):

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."   # Windows; open a new terminal afterwards
```

## Usage

```bash
python download_papers.py
python ingest.py
python ask.py "What problem does retrieval-augmented generation solve?"
```

Use a different Claude model with the `CLAUDE_MODEL` environment variable.

## Next steps

- [ ] Contextual Retrieval (Anthropic) — add context to each chunk before embedding
- [ ] Query rewriting / multi-query (RAG From Scratch part 5+)
- [ ] Re-ranking retrieved chunks
- [ ] A small evaluation set to measure whether changes actually help
