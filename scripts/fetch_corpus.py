"""Downloads the 18-paper golden-set corpus from arXiv into data/papers/.

Re-run anytime; skips files that already exist. arXiv IDs live in `app.core.corpus.CORPUS`,
the single source of truth shared with the golden-set validator.
Run standalone: `python scripts/fetch_corpus.py`.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.core.corpus import CORPUS

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"


def fetch_one(arxiv_id: str, title: str) -> tuple[str, str]:
    out_path = OUT_DIR / f"{arxiv_id}.pdf"
    if out_path.exists() and out_path.stat().st_size > 0:
        return arxiv_id, "already present"

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "rag-project-corpus-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        return arxiv_id, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return arxiv_id, f"error: {e.reason}"

    out_path.write_bytes(data)
    return arxiv_id, f"downloaded ({len(data) / 1024:.0f} KB)"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for arxiv_id, title in CORPUS.items():
        arxiv_id_result, status = fetch_one(arxiv_id, title)
        print(f"{arxiv_id_result:>12}  {title:<55} {status}")
        if status.startswith(("HTTP", "error")):
            failures.append(arxiv_id)
        time.sleep(1)  # be polite to arxiv.org

    if failures:
        print(f"\n{len(failures)} failed: {failures}", file=sys.stderr)
        return 1
    print(f"\nAll {len(CORPUS)} papers present in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
