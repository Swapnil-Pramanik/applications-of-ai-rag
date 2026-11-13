"""Format-specific parsers producing a common per-page text representation.

Every parser returns a list of `PageContent`, one entry per page, so that
downstream chunking can attach a page number to each chunk for citations.
Formats without a native page concept (TXT, MD) yield a single page; DOCX
has no reliable page boundary either (page breaks are a rendering artifact,
not stored per-run-of-text), so it is also treated as one page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf


@dataclass(frozen=True)
class PageContent:
    page_number: int  # 1-indexed
    text: str


class UnsupportedFormatError(ValueError):
    pass


def parse_pdf(path: Path) -> list[PageContent]:
    pages = []
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]  # pymupdf untyped
        for i, page in enumerate(doc):
            pages.append(PageContent(page_number=i + 1, text=page.get_text()))
    return pages


def parse_docx(path: Path) -> list[PageContent]:
    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs)
    return [PageContent(page_number=1, text=text)]


def parse_text(path: Path) -> list[PageContent]:
    text = path.read_text(encoding="utf-8")
    return [PageContent(page_number=1, text=text)]


_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".md": parse_text,
}


def parse_document(path: Path) -> list[PageContent]:
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFormatError(f"no parser for {suffix!r} (supported: {sorted(_PARSERS)})")
    return parser(path)
