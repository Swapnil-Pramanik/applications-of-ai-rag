"""Ties parse -> chunk -> embed -> index into one ingestion pass.

Deliberately DB-free. Persisting documents/chunks rows is the API layer's job
(Day 13) and mixing it in here would make the pipeline untestable without a
live Supabase and unusable from the eval harness, which ingests the corpus
without any tenant at all. What this module owns is the part that must stay
consistent between production ingestion and evaluation: identical parsing,
identical chunking, identical embedding, identical vector ids.

`vector_id` assignment is the one subtle piece. FAISS ids are int64 and must
be unique *within a tenant's index*; the caller supplies `start_vector_id` so
that ingesting a second document continues numbering rather than colliding
with the first. In production that watermark comes from
`max(chunks.vector_id)` for the tenant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.core.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import Embedder, Vector, get_embedder
from app.ingestion.index_builder import build_index
from app.ingestion.parsers import parse_document


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk plus everything needed to write its `chunks` row later."""

    vector_id: int
    chunk_index: int
    page_number: int
    text: str
    source_filename: str


@dataclass(frozen=True)
class IngestionResult:
    chunks: list[EmbeddedChunk]
    vectors: Vector
    page_count: int

    @property
    def next_vector_id(self) -> int:
        return self.chunks[-1].vector_id + 1 if self.chunks else 0


def ingest_document(
    path: Path,
    *,
    embedder: Embedder | None = None,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    start_vector_id: int = 0,
) -> IngestionResult:
    """Parse, chunk and embed one file. Does not touch FAISS or Postgres."""
    embedder = embedder or get_embedder()

    pages = parse_document(path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    return IngestionResult(
        chunks=[
            EmbeddedChunk(
                vector_id=start_vector_id + i,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                text=chunk.text,
                source_filename=path.name,
            )
            for i, chunk in enumerate(chunks)
        ],
        vectors=vectors,
        page_count=len(pages),
    )


def index_from_results(results: list[IngestionResult]) -> faiss.Index:
    """Build one tenant index from several ingested documents."""
    non_empty = [result for result in results if len(result.chunks) > 0]
    if not non_empty:
        return build_index(np.empty((0, 0), dtype="float32"), [])

    return build_index(
        np.vstack([result.vectors for result in non_empty]),
        [chunk.vector_id for result in non_empty for chunk in result.chunks],
    )
