# rag-from-papers

A minimal Retrieval-Augmented Generation (RAG) app that answers questions
about classic AI research papers, using either a local model (free, offline) or Claude.

I built this to learn how RAG works under the hood, while following LangChain's
*RAG From Scratch* series (parts 1–4). Everything is kept small and readable on
purpose: one file per stage, no frameworks.

## How it works

Indexing runs once, ahead of time. Retrieval and generation run on every question.

```mermaid
flowchart LR
    subgraph IDX["1 · Indexing — indexing.py (runs once)"]
        direction LR
        P["7 arXiv PDFs<br/>docs/"] --> L["Load text<br/>stop at References"]
        L --> C["Chunk<br/>1000 chars, 200 overlap"]
        C --> E1["Embed<br/>all-MiniLM-L6-v2"]
    end

    DB[("Chroma<br/>chroma_db/<br/>398 chunks")]
    E1 --> DB

    subgraph Q["Every question — ask.py"]
        direction LR
        U(["Question"]) --> E2["Embed<br/>same model"]
        E2 -->|"2 · Retrieval<br/>retrieval.py"| DB
        DB -->|"top 5 chunks"| G["3 · Generation<br/>generation.py"]
        U --> G
        G --> A(["Answer with<br/>citations [1]..[5]"])
    end
```

What happens during one `python ask.py "..."` call:

```mermaid
sequenceDiagram
    actor You
    participant ask as ask.py
    participant ret as retrieval.py
    participant db as Chroma
    participant gen as generation.py
    participant llm as LLM backend

    You->>ask: "What problem does RAG solve?"
    ask->>ret: retrieve(question)
    ret->>ret: embed question (384 numbers)
    ret->>db: query 5 nearest vectors
    db-->>ret: chunks + source + page + distance
    ret-->>ask: 5 chunks
    ask->>gen: generate(question, chunks)
    gen->>gen: build prompt with numbered sources
    gen->>llm: system prompt + sources + question
    llm-->>gen: answer citing [1]..[5]
    gen-->>ask: answer
    ask-->>You: sources list + answer
```

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

```mermaid
flowchart LR
    G["generation.py<br/>prompt with sources"] --> B{"LLM_BACKEND"}
    B -->|"ollama (default)"| O["Ollama<br/>llama3.2:3b<br/>local, free"]
    B -->|"claude-code"| CC["Claude Code CLI<br/>your subscription"]
    B -->|"claude"| API["Anthropic API<br/>ANTHROPIC_API_KEY"]
```

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

```mermaid
flowchart LR
    QS["eval/questions.json<br/>question, expected paper,<br/>required keywords"] --> R["retrieve top 5"]
    R --> M1["hit@5 and MRR<br/>was the expected paper found,<br/>and how high?"]
    R -->|"--answers"| GEN["generate answer"]
    GEN --> M2["keyword check<br/>does the answer contain<br/>the expected facts?"]
```

```bash
python eval/run_eval.py             # retrieval only: free, a few seconds
python eval/run_eval.py --answers   # also generate answers with LLM_BACKEND and keyword-check them
```

My baseline (fixed-size chunks, all-MiniLM-L6-v2, top 5, Claude via `claude-code`):

| Metric | Score |
|--------|-------|
| Retrieval hit@5 (right paper in top 5) | 13/13 |
| Retrieval MRR | 0.89 |
| Answers passing keyword check | 12/14 |

Both failures ask about the original RAG paper (Lewis et al.). The paper was retrieved, but not the passage holding the answer, and the model correctly replied that the sources don't say. Finding the right paper isn't enough; the right chunk has to be found too.

## What I learned

- **Bad chunks look relevant.** My first index had 796 chunks, and half were bibliography pages. Reference lists are full of keywords like "retrieval", so they ranked high for almost every question while containing no answers. Skipping everything after the "References" heading halved the index and fixed the results.
- **Retrieval and generation are independent.** The same 5 chunks came back no matter which LLM wrote the answer. Swapping models only changed the writing quality, which is why the backend is a single setting.
- **Grounding works, but it exposes retrieval gaps.** Telling the model to use only the sources made it say "I don't know" instead of guessing. Every remaining failure is a retrieval problem I can see and measure.
- **Measure before changing things.** One test question told me very little. A fixed set of 14 questions gives a baseline to compare every change against.

## Explainer

I made [`rag-explainer.html`](rag-explainer.html), an illustrated walkthrough of the three stages using the real numbers from this project, including what Chroma and the embedding model do and how to swap them. Download it and open it in a browser.

## Next steps

- [ ] Try a stronger embedding model (e.g. `BAAI/bge-small-en-v1.5`); the current one reads only 256 tokens per chunk
- [ ] Contextual Retrieval (Anthropic): add context to each chunk before embedding, then re-run the eval
- [ ] Re-ranking retrieved chunks
- [ ] Query rewriting / multi-query (RAG From Scratch part 5+)
- [x] A small evaluation set to measure whether changes actually help
