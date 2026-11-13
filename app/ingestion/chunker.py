"""Recursive character-based chunker.

Two-phase design, matching the standard recursive-splitter approach:

1. Split each page's text into small "atoms" using a priority list of
   separators (paragraph, then line, then sentence, then word, then raw
   character), recursing into a separator only when a piece still exceeds
   `chunk_size`.
2. Greedily merge atoms into chunks up to `chunk_size`, carrying the last
   `chunk_overlap` characters' worth of atoms into the start of the next
   chunk for continuity.

Doing overlap as part of the merge (not as a post-hoc concatenation) is
what keeps every emitted chunk within `chunk_size` even with overlap on.
Parameterized because chunk size/overlap is the axis the Day 9 eval sweep
tunes.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.ingestion.parsers import PageContent

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
JOIN_SEPARATOR = " "


@dataclass(frozen=True)
class Chunk:
    text: str
    page_number: int
    chunk_index: int  # position within the document, 0-indexed


def _split_into_atoms(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    separator, *remaining_separators = separators
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    atoms: list[str] = []
    for part in text.split(separator):
        if not part:
            continue
        if len(part) > chunk_size:
            atoms.extend(_split_into_atoms(part, chunk_size, remaining_separators))
        else:
            atoms.append(part)
    return atoms


def _merge_atoms(atoms: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def piece_len(atom: str) -> int:
        return len(atom) + (len(JOIN_SEPARATOR) if current else 0)

    for atom in atoms:
        if current and current_len + piece_len(atom) > chunk_size:
            chunks.append(JOIN_SEPARATOR.join(current))

            kept: list[str] = []
            kept_len = 0
            for prev_atom in reversed(current):
                extra = len(prev_atom) + (len(JOIN_SEPARATOR) if kept else 0)
                if kept_len + extra > chunk_overlap:
                    break
                kept.insert(0, prev_atom)
                kept_len += extra
            current = kept
            current_len = kept_len

        current_len += piece_len(atom)
        current.append(atom)

    if current:
        chunks.append(JOIN_SEPARATOR.join(current))
    return chunks


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    separators: list[str] | None = None,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    separators = separators if separators is not None else DEFAULT_SEPARATORS
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        atoms = _split_into_atoms(page.text, chunk_size, separators)
        for piece in _merge_atoms(atoms, chunk_size, chunk_overlap):
            stripped = piece.strip()
            if not stripped:
                continue
            chunks.append(Chunk(text=stripped, page_number=page.page_number, chunk_index=index))
            index += 1
    return chunks
