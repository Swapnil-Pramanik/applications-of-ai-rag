"""Checks every golden-set label against the source PDFs. Run: `make validate-golden`.

The golden set is the measuring instrument for every number this project
reports, so a wrong label does not look like a bug — it looks like a worse
retriever. This script is the machine half of the verification protocol: it
proves each supporting span is really on the page it claims, in the paper it
claims, in the text PyMuPDF actually extracts. The human half (does the answer
follow from the span, is the question fair) is the owner's pass, tracked by the
`verified` flag, which this script only counts and never sets.

Exit code is non-zero if any label fails, so CI can gate on it once the corpus
is fetchable there.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from app.core.corpus import CORPUS, title_for
from app.ingestion.parsers import parse_pdf
from eval.harness.golden_set import QUESTIONS_PATH, GoldenQuestion, load_golden_set, normalize

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "papers"
VALID_TYPES = {"factual", "numeric", "comparative", "multi_hop"}
VALID_RELEVANCE = {"any", "all"}
MAX_SPAN_CHARS = 200  # longer spans risk straddling a chunk boundary; see golden_set.py


def _check_schema(questions: list[GoldenQuestion]) -> list[str]:
    errors = []
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            errors.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.type not in VALID_TYPES:
            errors.append(f"{q.id}: unknown type {q.type!r}")
        if q.relevance not in VALID_RELEVANCE:
            errors.append(f"{q.id}: unknown relevance {q.relevance!r}")
        if not q.supporting_spans:
            errors.append(f"{q.id}: no supporting spans")
        if q.relevance == "all" and len(q.supporting_spans) < 2:
            errors.append(f"{q.id}: relevance 'all' with a single span — use 'any'")
        for span in q.supporting_spans:
            if span.source_filename.removesuffix(".pdf") not in CORPUS:
                errors.append(f"{q.id}: {span.source_filename} is not in the corpus")
            if len(span.span) > MAX_SPAN_CHARS:
                errors.append(f"{q.id}: span of {len(span.span)} chars exceeds {MAX_SPAN_CHARS}")
    return errors


def _check_spans(questions: list[GoldenQuestion], corpus_dir: Path) -> list[str]:
    """Every span must appear, normalized, in the extracted text of its stated page."""
    errors = []
    page_cache: dict[str, list[str]] = {}

    for q in questions:
        for span in q.supporting_spans:
            filename = span.source_filename
            if filename not in page_cache:
                path = corpus_dir / filename
                if not path.exists():
                    errors.append(f"{q.id}: {path} not found — run `make corpus`")
                    page_cache[filename] = []
                    continue
                page_cache[filename] = [normalize(p.text) for p in parse_pdf(path)]

            pages = page_cache[filename]
            if not pages:
                continue
            if not 1 <= span.page_number <= len(pages):
                errors.append(
                    f"{q.id}: {filename} has {len(pages)} pages, span claims p{span.page_number}"
                )
                continue

            needle = normalize(span.span)
            if needle in pages[span.page_number - 1]:
                continue

            elsewhere = [i + 1 for i, text in enumerate(pages) if needle in text]
            hint = f" (found on p{elsewhere})" if elsewhere else ""
            errors.append(
                f"{q.id}: span not on p{span.page_number} of {title_for(filename)}{hint}: "
                f"{span.span[:70]!r}"
            )
    return errors


def _summarize(questions: list[GoldenQuestion]) -> str:
    by_type = Counter(q.type for q in questions)
    by_paper = Counter(f for q in questions for f in q.source_filenames)
    verified = sum(q.verified for q in questions)
    lines = [
        f"{len(questions)} questions, {sum(len(q.supporting_spans) for q in questions)} spans",
        "  by type:  " + ", ".join(f"{t}={n}" for t, n in sorted(by_type.items())),
        f"  papers covered: {len(by_paper)}/{len(CORPUS)}",
        f"  human-verified: {verified}/{len(questions)}",
    ]
    uncovered = sorted(set(CORPUS) - {f.removesuffix(".pdf") for f in by_paper})
    if uncovered:
        lines.append("  no questions yet: " + ", ".join(title_for(u) for u in uncovered))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="skip PDF span checks (for environments without the corpus)",
    )
    args = parser.parse_args(argv)

    questions = load_golden_set(args.questions)
    print(_summarize(questions))

    errors = _check_schema(questions)
    if not args.schema_only:
        errors += _check_spans(questions, args.corpus_dir)

    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("\nall labels check out against the source PDFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
