"""Download a small set of public AI papers from arXiv into docs/.

The PDFs are not committed to the repo (see .gitignore); anyone who clones
it runs this script to fetch them.
"""

import time
import urllib.request
from pathlib import Path

DOCS_DIR = Path("docs")

# arXiv ID -> short file name
PAPERS = {
    "1706.03762": "attention-is-all-you-need",
    "1810.04805": "bert",
    "1908.10084": "sentence-bert",
    "2005.11401": "rag-lewis-2020",
    "2201.11903": "chain-of-thought",
    "2210.03629": "react",
    "2312.10997": "rag-survey",
}


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    for arxiv_id, name in PAPERS.items():
        target = DOCS_DIR / f"{name}.pdf"
        if target.exists():
            print(f"skip  {target} (already downloaded)")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"fetch {url} -> {target}")
        request = urllib.request.Request(url, headers={"User-Agent": "rag-from-papers/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        time.sleep(3)  # be polite to arXiv


if __name__ == "__main__":
    main()
