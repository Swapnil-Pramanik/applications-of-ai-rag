"""Cross-encoder reranking tests.

All offline — the cross-encoder is stubbed, because what needs pinning is the
wiring, not MiniLM's opinions. Two properties carry real weight: the stage must
be handed a *wide* candidate pool (asking the base retriever for `k` would make
reranking a no-op that still costs a model load), and the ordering must be
deterministic (a tie broken differently between runs silently perturbs every
Recall@k the harness reports).
"""

from __future__ import annotations

import pytest

from app.retrieval.base import RetrievedChunk
from app.retrieval.reranker import Reranker, RerankingRetriever


def chunk(vector_id: int, text: str = "", score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        vector_id=vector_id,
        text=text or f"chunk {vector_id}",
        page_number=1,
        chunk_index=vector_id,
        source_filename="paper.pdf",
        score=score,
    )


class StubBase:
    """Records the k it was asked for, so the widening can be asserted."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks
        self.asked_for: list[int] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.asked_for.append(k)
        return self._chunks[:k]


class StubReranker(Reranker):
    """Scores by a lookup table keyed on chunk text."""

    def __init__(self, scores: dict[str, float]):
        super().__init__(model=object())  # never touched
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [self._scores.get(text, 0.0) for text in texts]


# --- the candidate pool ----------------------------------------------------


def test_the_base_is_asked_for_the_full_candidate_pool_not_just_k() -> None:
    """Reranking can only reorder what it is given. Asking for k is a no-op."""
    base = StubBase([chunk(i) for i in range(20)])
    retriever = RerankingRetriever(base, StubReranker({}), candidate_k=20)

    retriever.retrieve("q", k=5)

    assert base.asked_for == [20]


def test_a_k_larger_than_the_candidate_pool_widens_the_pool() -> None:
    """Otherwise the caller silently gets fewer results than they asked for."""
    base = StubBase([chunk(i) for i in range(50)])
    retriever = RerankingRetriever(base, StubReranker({}), candidate_k=20)

    assert len(retriever.retrieve("q", k=30)) == 30
    assert base.asked_for == [30]


# --- reordering ------------------------------------------------------------


def test_the_cross_encoder_reorders_the_base_ranking() -> None:
    """The whole point: a chunk the base ranked last can win on true relevance."""
    base = StubBase([chunk(1, "unrelated"), chunk(2, "nearby"), chunk(3, "the answer")])
    reranker = StubReranker({"the answer": 9.0, "nearby": 1.0, "unrelated": -3.0})

    ranked = RerankingRetriever(base, reranker, candidate_k=3).retrieve("q", k=3)

    assert [c.vector_id for c in ranked] == [3, 2, 1]


def test_the_returned_score_is_the_cross_encoder_logit_not_the_base_score() -> None:
    """Downstream (the Day 11 abstention gate) calibrates against this number."""
    base = StubBase([chunk(1, "the answer", score=0.016)])
    reranker = StubReranker({"the answer": 7.5})

    ranked = RerankingRetriever(base, reranker, candidate_k=1).retrieve("q", k=1)

    assert ranked[0].score == pytest.approx(7.5)


def test_ties_are_broken_by_vector_id_so_the_ranking_is_deterministic() -> None:
    base = StubBase([chunk(3, "a"), chunk(1, "a"), chunk(2, "a")])
    reranker = StubReranker({"a": 1.0})

    ranked = RerankingRetriever(base, reranker, candidate_k=3).retrieve("q", k=3)

    assert [c.vector_id for c in ranked] == [1, 2, 3]


def test_the_query_reaches_the_cross_encoder_paired_with_every_candidate() -> None:
    base = StubBase([chunk(1, "alpha"), chunk(2, "beta")])
    reranker = StubReranker({})

    RerankingRetriever(base, reranker, candidate_k=2).retrieve("why?", k=2)

    assert reranker.calls == [("why?", ["alpha", "beta"])]


# --- degenerate inputs -----------------------------------------------------


def test_a_non_positive_k_returns_nothing_without_loading_a_model() -> None:
    base = StubBase([chunk(1)])

    assert RerankingRetriever(base, StubReranker({})).retrieve("q", k=0) == []
    assert base.asked_for == []


def test_an_empty_candidate_pool_short_circuits_before_scoring() -> None:
    reranker = StubReranker({})

    assert RerankingRetriever(StubBase([]), reranker).retrieve("q", k=5) == []
    assert reranker.calls == []


def test_scoring_no_texts_returns_no_scores_without_calling_the_model() -> None:
    """Guards the real `Reranker`, whose `model` property would load MiniLM."""
    assert Reranker(model=None, model_name="never-loaded").score("q", []) == []
