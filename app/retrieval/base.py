"""The one interface every retrieval strategy implements.

Day 8's ablation has to run dense-only, sparse-only and hybrid over the same
golden set and report the deltas. That only produces a fair comparison if the
harness can swap strategies without changing anything else, so all three are
the same shape: `retrieve(query, k) -> list[RetrievedChunk]`, best first.

`score` is deliberately **not** comparable across implementations — cosine
similarity for dense, BM25's unbounded term-weight sum for sparse, a
reciprocal-rank sum for hybrid. It is meaningful for ranking within one
retriever's results and for the confidence gate (Day 11), which is calibrated
against whichever retriever is actually deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.retrieval.chunk_store import StoredChunk


@dataclass(frozen=True)
class RetrievedChunk:
    vector_id: int
    text: str
    page_number: int
    chunk_index: int
    source_filename: str
    score: float

    @classmethod
    def from_stored(cls, chunk: StoredChunk, score: float) -> RetrievedChunk:
        return cls(
            vector_id=chunk.vector_id,
            text=chunk.text,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            source_filename=chunk.source_filename,
            score=score,
        )


@runtime_checkable
class Retriever(Protocol):
    # Whether `score` means the same thing from one query to the next.
    #
    # False for everything that scores query and passage separately: cosine
    # drifts with phrasing, BM25's magnitude depends on corpus statistics, and
    # RRF sums reciprocal *ranks*, which carry no relevance signal at all. Only
    # the cross-encoder reads the pair together, so only it produces a number a
    # fixed threshold can be compared against.
    #
    # The abstention gate refuses to run behind a retriever that says False —
    # see `pipeline.QueryPipeline`. Declared as a property of the retriever
    # rather than checked with `isinstance` because the thing that matters is
    # what the scores mean, not which class produced them.
    emits_calibrated_scores: bool = False

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return at most `k` chunks for `query`, highest score first."""
        ...
