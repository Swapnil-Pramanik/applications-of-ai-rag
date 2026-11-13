"""Reciprocal rank fusion tests.

The arithmetic is trivial; what's worth pinning is the *behaviour* the choice
of RRF was made for — that agreement between two retrievers beats a single
retriever's top hit — plus determinism, since a tie broken differently
between runs silently perturbs Recall@k in the eval harness.
"""

from __future__ import annotations

import pytest

from app.retrieval.base import RetrievedChunk
from app.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion


def chunk(vector_id: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        vector_id=vector_id,
        text=f"chunk {vector_id}",
        page_number=1,
        chunk_index=vector_id,
        source_filename="paper.pdf",
        score=score,
    )


class StubRetriever:
    def __init__(self, vector_ids: list[int]):
        self.vector_ids = vector_ids
        self.seen: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.seen.append((query, k))
        return [chunk(vector_id) for vector_id in self.vector_ids[:k]]


# --- fusion arithmetic -----------------------------------------------------


def test_a_document_ranked_well_by_both_beats_one_ranked_first_by_one() -> None:
    """The entire reason this system fuses instead of picking a winner.

    Doc 2 is second in both lists; doc 1 is first in one and absent from the
    other. With rrf_k=60 the corroborated document wins.
    """
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2)], [chunk(3), chunk(2)]], k=3, rrf_k=60)

    assert [hit.vector_id for hit in fused] == [2, 1, 3]


def test_a_low_rrf_k_makes_fusion_winner_take_all() -> None:
    """Same inputs, k=0: rank 1 is now worth twice rank 2, so doc 1 wins."""
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2)], [chunk(3), chunk(2)]], k=3, rrf_k=0)

    assert fused[0].vector_id == 1


def test_the_fused_score_is_the_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([[chunk(1)], [chunk(1)]], k=1, rrf_k=60)

    assert fused[0].score == pytest.approx(2 / 61)


def test_the_original_retriever_score_is_replaced_by_the_fused_one() -> None:
    fused = reciprocal_rank_fusion([[chunk(1, score=9.9)]], k=1, rrf_k=60)

    assert fused[0].score == pytest.approx(1 / 61)


def test_chunk_metadata_survives_fusion() -> None:
    fused = reciprocal_rank_fusion([[chunk(7)]], k=1, rrf_k=60)

    assert (fused[0].text, fused[0].page_number, fused[0].source_filename) == (
        "chunk 7",
        1,
        "paper.pdf",
    )


def test_ties_are_broken_deterministically_by_vector_id() -> None:
    fused = reciprocal_rank_fusion([[chunk(9), chunk(4)], [chunk(4), chunk(9)]], k=2, rrf_k=60)

    assert [hit.vector_id for hit in fused] == [4, 9]


def test_k_truncates_the_fused_list() -> None:
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2), chunk(3)]], k=2, rrf_k=60)

    assert len(fused) == 2


def test_fusing_nothing_returns_nothing() -> None:
    assert reciprocal_rank_fusion([[], []], k=5, rrf_k=60) == []


# --- HybridRetriever -------------------------------------------------------


def test_sub_retrievers_are_queried_over_the_wider_candidate_pool() -> None:
    dense, sparse = StubRetriever([1, 2]), StubRetriever([3, 4])
    HybridRetriever([dense, sparse], candidate_k=20, rrf_k=60).retrieve("query", k=5)

    assert dense.seen == sparse.seen == [("query", 20)]


def test_hybrid_returns_only_k_even_though_it_fetched_candidate_k() -> None:
    dense, sparse = StubRetriever([1, 2, 3]), StubRetriever([4, 5, 6])

    fused = HybridRetriever([dense, sparse], candidate_k=20, rrf_k=60).retrieve("query", k=2)

    assert len(fused) == 2


def test_a_single_retriever_degrades_to_that_retriever_s_ordering() -> None:
    """Makes the Day 8 ablation honest: dense-only through the hybrid path is
    the same ranking as dense-only alone, so the comparison isolates fusion."""
    dense = StubRetriever([5, 6, 7])

    fused = HybridRetriever([dense], candidate_k=20, rrf_k=60).retrieve("query", k=3)

    assert [hit.vector_id for hit in fused] == [5, 6, 7]


def test_a_hybrid_with_no_retrievers_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="at least one retriever"):
        HybridRetriever([])


def test_non_positive_k_short_circuits_without_querying_sub_retrievers() -> None:
    dense = StubRetriever([1, 2])

    assert HybridRetriever([dense]).retrieve("query", k=0) == []
    assert dense.seen == []


def test_defaults_come_from_settings() -> None:
    from app.core.config import settings

    dense = StubRetriever([1, 2])
    HybridRetriever([dense]).retrieve("query", k=1)

    assert dense.seen == [("query", settings.candidate_k)]
