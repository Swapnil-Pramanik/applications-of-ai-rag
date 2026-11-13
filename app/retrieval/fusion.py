"""Reciprocal Rank Fusion over several retrievers.

**Why rank fusion and not score fusion.** Dense cosine similarity lives in a
bounded [-1, 1] and clusters tightly near the top; BM25 is unbounded and its
absolute magnitude depends on corpus statistics, so the same score means
different things for different tenants and different queries. Combining them
numerically requires normalizing two distributions that have no shared unit —
min-max over the candidate list is the usual hack, and it makes the fused
score depend on how many candidates happened to come back.

RRF sidesteps all of that by throwing the scores away and keeping only the
ordering:

    score(d) = Σ over retrievers of 1 / (rrf_k + rank(d))

with `rank` 1-based. No normalization, no per-corpus tuning, one parameter.

**What `rrf_k` does.** It damps the advantage of a rank-1 hit. At k=60
(Cormack, Clarke & Buettcher 2009, and the value everyone inherited) the gap
between rank 1 and rank 2 is small, so a document ranked moderately well by
*both* retrievers beats one ranked first by a single retriever and missed
entirely by the other. That is precisely the behaviour we want: agreement
across two different notions of relevance is stronger evidence than one
retriever's confidence. Lowering `rrf_k` toward 0 makes it winner-take-all.
It is config, not a constant, because Day 8 sweeps it.

The fused `score` is a reciprocal-rank sum, roughly bounded by
`n_retrievers / (rrf_k + 1)` — it is a ranking quantity and carries no
similarity meaning, so the Day 11 confidence gate must be calibrated against
it, not against a cosine threshold.
"""

from __future__ import annotations

from dataclasses import replace

from app.core.config import settings
from app.retrieval.base import RetrievedChunk, Retriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int,
    rrf_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, best first, keeping at most `k`."""
    rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.vector_id] = scores.get(chunk.vector_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(chunk.vector_id, chunk)

    # Ties are broken by vector_id so a fused ranking is deterministic across
    # runs — an eval harness that reorders ties silently perturbs Recall@k.
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [replace(chunks[vector_id], score=score) for vector_id, score in ordered[:k]]


class HybridRetriever:
    """Runs each sub-retriever over a wide candidate pool, then fuses.

    `candidate_k` is intentionally larger than the `k` returned (20 -> 5 in
    the default config). Fusion can only rerank what it was given, so a
    document has to appear in at least one retriever's candidate list to be
    recoverable at all; a narrow pool makes RRF's agreement signal meaningless
    because there is nothing to agree about.
    """

    # RRF sums reciprocal *ranks*. A rank-1 hit scores ~1/61 whether the
    # search was perfect or catastrophic, so this carries no relevance signal.
    emits_calibrated_scores = False

    def __init__(
        self,
        retrievers: list[Retriever],
        candidate_k: int | None = None,
        rrf_k: int | None = None,
    ):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = retrievers
        self._candidate_k = candidate_k if candidate_k is not None else settings.candidate_k
        self._rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        ranked_lists = [
            retriever.retrieve(query, self._candidate_k) for retriever in self._retrievers
        ]
        return reciprocal_rank_fusion(ranked_lists, k, self._rrf_k)
