"""Where retrievers get the text behind a vector id.

Retrieval only ever knows about `vector_id` — that is the int64 FAISS returns
and the int64 the BM25 index carries alongside its corpus. Turning those ids
back into displayable, citable chunks is a separate concern, and it has two
very different implementations:

- **In production** the source of truth is the `chunks` table in Postgres.
- **In the eval harness and in tests** there is no tenant and no database at
  all — the corpus is ingested in-process (see `app.ingestion.pipeline`).

Both are hidden behind the `ChunkStore` protocol so retrieval code is
identical in either case. That matters more than it looks: the whole pitch of
this project is that the measured numbers describe the deployed system, which
is only true if the eval harness exercises the same retrieval path production
does.

**There is deliberately no `PostgresChunkStore`.** This module used to promise
one — a store that resolves chunk text by vector id on demand. Building the API
(Day 13) showed that to be the wrong shape: retrieval resolves 20 candidates
per query, so a lazy store is an N+1 across the network *inside* the hot path,
against a retrieval budget measured at 0.17s p95.

The text has to be resident anyway — the cross-encoder scores it, citations
quote it, BM25 indexes it — so `api/deps.py` reads a tenant's chunks once, in
one query, into an `InMemoryChunkStore` cached beside their FAISS and BM25
indexes. Production and eval therefore use the *same* implementation, differing
only in where the rows came from.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StoredChunk:
    """One chunk's text and the metadata a citation is assembled from."""

    vector_id: int
    text: str
    page_number: int
    chunk_index: int
    source_filename: str


@runtime_checkable
class ChunkStore(Protocol):
    def get_many(self, vector_ids: Iterable[int]) -> dict[int, StoredChunk]:
        """Look up chunks by vector id.

        Returns a mapping rather than a list because ids that no longer exist
        (a document deleted after its vectors were indexed) must be droppable
        by the caller without silently shifting the result ordering.
        """
        ...


class InMemoryChunkStore:
    """A dict-backed `ChunkStore` for the eval harness and unit tests."""

    def __init__(self, chunks: Iterable[StoredChunk]):
        self._chunks = {chunk.vector_id: chunk for chunk in chunks}

    def __len__(self) -> int:
        return len(self._chunks)

    def get_many(self, vector_ids: Iterable[int]) -> dict[int, StoredChunk]:
        return {
            vector_id: self._chunks[vector_id]
            for vector_id in vector_ids
            if vector_id in self._chunks
        }
