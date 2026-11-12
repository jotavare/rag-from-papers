# rag-from-papers

A minimal Retrieval-Augmented Generation (RAG) app that answers questions
about classic AI research papers, using either a local model (free, offline) or Claude.

I built this to learn how RAG works under the hood, while following LangChain's
*RAG From Scratch* series (parts 1–4). Everything is kept small and readable on
purpose: one file per stage, no frameworks.

**How RAG works, explained with this project's numbers:** [jotavare.github.io/rag-from-papers](https://jotavare.github.io/rag-from-papers/)

## Project structure

```
rag-from-papers/
├── rag/                    # the pipeline
│   ├── config.py           # paths and shared settings
│   ├── download_papers.py  # setup: fetch the papers from arXiv
│   ├── indexing.py         # stage 1: PDFs → chunks → embeddings → Chroma
│   ├── planning.py         # optional: LLM picks a paper and search phrase first
│   ├── retrieval.py        # stage 2: question → 5 most similar chunks
│   ├── generation.py       # stage 3: chunks + question → LLM → cited answer
│   └── ask.py              # runs retrieval, then generation
├── eval/
│   ├── questions.json      # 14 questions with known answers
│   └── run_eval.py         # scores retrieval and answers
├── docs/
│   └── index.html          # the explainer page (GitHub Pages)
└── data/                   # created locally, not committed
    ├── papers/             # downloaded PDFs
    └── chroma_db/          # vector database
```

## How it works

Indexing runs once, ahead of time. Retrieval and generation run on every question.

```mermaid
flowchart LR
    subgraph IDX["1 · Indexing — rag/indexing.py (runs once)"]
        direction LR
        P["7 arXiv PDFs<br/>data/papers/"] --> L["Load text<br/>stop at References"]
        L --> C["Chunk<br/>1000 chars, 200 overlap"]
        C --> E1["Embed<br/>all-MiniLM-L6-v2"]
    end

    DB[("Chroma<br/>data/chroma_db/<br/>398 chunks")]
    E1 --> DB

    subgraph Q["Every question — rag/ask.py"]
        direction LR
        U(["Question"]) --> E2["Embed<br/>same model"]
        E2 -->|"2 · Retrieval<br/>rag/retrieval.py"| DB
        DB -->|"top 5 chunks"| G["3 · Generation<br/>rag/generation.py"]
        U --> G
        G --> A(["Answer with<br/>citations [1]..[5]"])
    end
```

What happens during one `python -m rag.ask "..."` call:

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

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m rag.download_papers   # 7 papers from arXiv into data/papers/
python -m rag.indexing          # builds data/chroma_db/
```

The papers are downloaded rather than committed, because arXiv's default license doesn't allow redistributing them.

## Usage

```bash
python -m rag.ask "What problem does retrieval-augmented generation solve?"
python -m rag.ask --plan "What retriever does the original RAG model use?"
```

### Planning the search (`--plan`)

Plain retrieval only compares numbers, so it can't act on hints in the
question like "proposed by Lewis et al.". With `--plan`, the LLM reads the
question first and returns a search plan: which paper to search (or all of
them) and a better search phrase. Retrieval then searches only that paper.

```mermaid
flowchart LR
    Q(["Question"]) --> PL["planning.py<br/>LLM picks paper + search phrase"]
    PL -->|"paper filter"| R["retrieval.py<br/>search only that paper"]
    R --> G["generation.py"]
    Q --> G
    G --> A(["Answer"])
```

It uses the same `LLM_BACKEND` as generation, so each question costs two LLM calls instead of one.

### Changing the embedding model

Set `EMBED_MODEL` (default `all-MiniLM-L6-v2`), then rebuild the index. Retrieval
refuses to run if the index was built with a different model.

```bash
$env:EMBED_MODEL = "BAAI/bge-small-en-v1.5"
python -m rag.indexing
```

### Choosing the model that writes answers

Set `LLM_BACKEND` (retrieval is the same for all of them):

```mermaid
flowchart LR
    G["generation.py<br/>prompt with sources"] --> B{"LLM_BACKEND"}
    B -->|"ollama (default)"| O["Ollama<br/>llama3.2:3b<br/>local, free"]
    B -->|"claude-code"| CC["Claude Code CLI<br/>Claude subscription"]
    B -->|"claude"| API["Anthropic API<br/>ANTHROPIC_API_KEY"]
```

| Backend | Needs | Notes |
|---------|-------|-------|
| `ollama` (default) | [Ollama](https://ollama.com) + `ollama pull llama3.2:3b` | Free and offline; ~1 min per answer on CPU. Change model with `OLLAMA_MODEL` |
| `claude-code` | [Claude Code](https://claude.com/claude-code) signed in to a Claude subscription | Counts against the plan's usage limits; no API key |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic API; change model with `CLAUDE_MODEL` |

```bash
# PowerShell: pick another backend for this terminal
$env:LLM_BACKEND = "claude-code"
python -m rag.ask "How does ReAct combine reasoning and acting?"
```

## Evaluation

`eval/questions.json` holds 14 questions with known answers: 13 answerable from a specific paper, plus one off-topic question the app should refuse.

```bash
python -m eval.run_eval                    # retrieval only: free, a few seconds
python -m eval.run_eval --answers          # also generate answers with LLM_BACKEND and keyword-check them
python -m eval.run_eval --plan --answers   # same, with the planning step
```

```mermaid
flowchart LR
    QS["eval/questions.json<br/>question, expected paper,<br/>required keywords"] --> R["retrieve top 5"]
    R --> M1["hit@5 and MRR<br/>was the expected paper found,<br/>and how high?"]
    R -->|"--answers"| GEN["generate answer"]
    GEN --> M2["keyword check<br/>does the answer contain<br/>the expected facts?"]
```

### Results

All runs use 1000-character chunks with 200 overlap, top 5, and Claude (`claude-code`) for answers.

| Run | Embedding model | Planning | hit@5 | MRR | Answers |
|-----|-----------------|----------|-------|-----|---------|
| Baseline | all-MiniLM-L6-v2 | – | 13/13 | 0.89 | 12/14 |
| Bigger context model | bge-small-en-v1.5 | – | 11/13 | 0.85 | not run |
| Search planning | all-MiniLM-L6-v2 | Claude | 13/13 | **1.00** | **13/14** |

**Baseline.** Both answer failures ask about the original RAG paper (Lewis et al.). The paper was retrieved, but not the passage holding the answer, and the model correctly replied that the sources don't say. Finding the right paper isn't enough; the right chunk has to be found too.

**bge-small-en-v1.5.** Despite reading 512 tokens instead of 256 and scoring higher on public benchmarks, it did worse on my papers: the Lewis paper dropped out of the top 5 for both of its questions. Adding the query instruction the model recommends changed nothing. I kept MiniLM as the default.

**Search planning.** Both Lewis questions now pass, and the expected paper ranks first for every question. One question that used to pass now fails: the planner narrowed "What problems does RAG address?" to the survey and rewrote it into a search phrase that pulled different passages. An earlier run of the same question chose "all papers" and passed, so the planner isn't fully consistent between runs.

## What I learned

- **Bad chunks look relevant.** My first index had 796 chunks, and half were bibliography pages. Reference lists are full of keywords like "retrieval", so they ranked high for almost every question while containing no answers. Skipping everything after the "References" heading halved the index and fixed the results.
- **Retrieval and generation are independent.** The same 5 chunks came back no matter which LLM wrote the answer. Swapping models only changed the writing quality, which is why the backend is a single setting.
- **Grounding works, but it exposes retrieval gaps.** Telling the model to use only the sources made it say "I don't know" instead of guessing. Every remaining failure is a retrieval problem I can see and measure.
- **Measure before changing things.** One test question told me very little. A fixed set of 14 questions gives a baseline to compare every change against.
- **"Better" models aren't better for everything.** A newer embedding model with a bigger input limit did worse on my papers. Only my own eval showed that.
- **Letting the LLM plan the search fixed what embeddings couldn't.** Embeddings can't act on "proposed by Lewis et al."; an LLM can. The cost is an extra call per question and less predictable behaviour.

## Next steps

- [x] Try a stronger embedding model (`bge-small-en-v1.5`: worse on these papers)
- [x] Plan the search with an LLM before retrieval
- [ ] Larger test set from Vectara's Open RAG Bench (CC BY-NC 4.0), with section-level labels
- [ ] Contextual Retrieval (Anthropic): add context to each chunk before embedding, then re-run the eval
- [ ] Re-ranking retrieved chunks
- [ ] Query rewriting / multi-query (RAG From Scratch part 5+)
- [x] A small evaluation set to measure whether changes actually help
