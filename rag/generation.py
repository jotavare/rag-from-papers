"""RAG step 3 — Generation (RAG From Scratch, Part 4).

retrieved chunks + question -> prompt -> LLM -> answer

The LLM is swappable; RAG doesn't care which model writes the answer.
Choose with the LLM_BACKEND environment variable:
  ollama       (default) a local model via Ollama — free, no API key
  claude-code  Claude via the Claude Code CLI — uses your Claude subscription
  claude       Claude via the Anthropic API — needs ANTHROPIC_API_KEY
"""

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

BACKEND = os.environ.get("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = "http://localhost:11434/api/chat"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You answer questions about AI research papers.
Use ONLY the numbered sources provided. Cite them like [1], [2].
If the sources don't contain the answer, say you don't know."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Number the chunks so the model can cite them."""
    sources = "\n\n".join(
        f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def ask_ollama(prompt: str) -> str:
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["message"]["content"]


def ask_claude(prompt: str) -> str:
    import anthropic  # only needed for this backend

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def ask_claude_code(prompt: str) -> str:
    """Run Claude Code in non-interactive mode (-p) with tools disabled,
    so it only reads the prompt and answers."""
    exe = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")
    result = subprocess.run(
        [exe, "-p", "--model", "sonnet", "--tools", "", "--system-prompt", SYSTEM_PROMPT],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    if result.returncode != 0:
        raise SystemExit(f"claude failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


BACKENDS = {"ollama": ask_ollama, "claude-code": ask_claude_code, "claude": ask_claude}


def generate(question: str, chunks: list[dict]) -> str:
    if BACKEND not in BACKENDS:
        raise SystemExit(f"Unknown LLM_BACKEND {BACKEND!r}; choose from {list(BACKENDS)}")
    return BACKENDS[BACKEND](build_prompt(question, chunks))
