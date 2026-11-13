"""Retrieval-metric tests.

These are the arithmetic behind every number this project reports, so what is
pinned here is the behaviour that would silently produce a *plausible wrong
answer*: the `any` vs `all` distinction that makes multi-hop questions honest,
and nDCG's ideal denominator, which is the piece most likely to be quietly
wrong because a too-generous denominator inflates every score at once.
"""

from __future__ import annotations

from math import log2

import pytest

from eval.harness.golden_set import GoldenQuestion, SupportingSpan
from eval.harness.metrics import (
    QuestionScore,
    aggregate,
    dcg,
    first_satisfied_rank,
    is_relevant_chunk,
    ndcg_at_k,
    relevant_chunk_count,
)


def span(text: str, page: int = 1) -> SupportingSpan:
    return SupportingSpan(source_filename="paper.pdf", page_number=page, span=text)


def question(
    *spans: SupportingSpan, relevance: str = "any", qtype: str = "factual"
) -> GoldenQuestion:
    return GoldenQuestion(
        id="q1",
        question="why?",
        answer="because",
        type=qtype,  # type: ignore[arg-type]
        relevance=relevance,  # type: ignore[arg-type]
        supporting_spans=spans,
        verified=False,
    )


# --- first_satisfied_rank --------------------------------------------------


def test_rank_is_where_the_only_required_span_appears() -> None:
    q = question(span("attention is all you need"))

    assert first_satisfied_rank(q, ["irrelevant", "ATTENTION IS ALL YOU NEED", "also no"]) == 2


def test_rank_is_none_when_the_span_never_appears() -> None:
    assert first_satisfied_rank(question(span("missing")), ["a", "b"]) is None


def test_relevance_all_waits_for_the_last_span_not_the_first() -> None:
    """The multi-hop correctness property. `any` would score this a rank-1 hit.

    Half the evidence for a multi-hop question is not an answer, and a metric
    that says otherwise makes the hardest question type look like the easiest.
    """
    q = question(span("first fact"), span("second fact"), relevance="all")
    texts = ["first fact here", "unrelated", "and the second fact"]

    assert first_satisfied_rank(q, texts) == 3
    assert first_satisfied_rank(question(span("first fact"), span("second fact")), texts) == 1


def test_rank_is_none_for_an_empty_retrieved_list() -> None:
    assert first_satisfied_rank(question(span("x")), []) is None


# --- relevance of individual chunks ---------------------------------------


def test_a_chunk_carrying_any_one_span_is_relevant() -> None:
    q = question(span("alpha"), span("beta"), relevance="all")

    assert is_relevant_chunk(q, "text mentioning beta only")


def test_relevant_chunk_count_scans_the_whole_corpus() -> None:
    q = question(span("alpha"))

    assert relevant_chunk_count(q, ["alpha here", "nothing", "alpha again"]) == 2


# --- nDCG ------------------------------------------------------------------


def test_dcg_discounts_by_log_of_rank() -> None:
    assert dcg([1.0, 1.0]) == pytest.approx(1 / log2(2) + 1 / log2(3))


def test_ndcg_is_one_when_every_relevant_chunk_is_ranked_first() -> None:
    q = question(span("alpha"))

    assert ndcg_at_k(q, ["alpha", "no", "no"], k=3, n_relevant=1) == pytest.approx(1.0)


def test_ndcg_penalises_a_relevant_chunk_ranked_lower() -> None:
    q = question(span("alpha"))
    perfect = ndcg_at_k(q, ["alpha", "no"], k=2, n_relevant=1)
    demoted = ndcg_at_k(q, ["no", "alpha"], k=2, n_relevant=1)

    assert 0 < demoted < perfect


def test_ndcg_ideal_is_capped_at_k_not_at_the_corpus_count() -> None:
    """With 10 relevant chunks and k=2, retrieving 2 of them is still a perfect run.

    Dividing by the un-capped ideal would make a perfect top-2 score ~0.3 and
    make every configuration look worse than it is.
    """
    q = question(span("alpha"))

    assert ndcg_at_k(q, ["alpha", "alpha"], k=2, n_relevant=10) == pytest.approx(1.0)


def test_ndcg_is_zero_when_no_chunk_in_the_corpus_is_relevant() -> None:
    """A labelling failure scores 0 rather than being skipped, so run-to-run
    averages always share a denominator."""
    assert ndcg_at_k(question(span("alpha")), ["alpha"], k=1, n_relevant=0) == 0.0


# --- aggregation -----------------------------------------------------------


def score(rank: int | None, ndcg: float = 0.5, qtype: str = "factual") -> QuestionScore:
    return QuestionScore(
        question_id="q",
        question_type=qtype,
        rank=rank,
        ndcg=ndcg,
        n_relevant=1,
        retrieved_sources=("paper.pdf",),
    )


def test_recall_counts_only_hits_within_the_cutoff() -> None:
    agg = aggregate([score(1), score(7), score(None)], k=10, cutoffs=[5, 10])

    assert agg.recall_at_k[5] == pytest.approx(1 / 3)
    assert agg.recall_at_k[10] == pytest.approx(2 / 3)


def test_mrr_uses_the_reciprocal_of_the_true_rank_within_the_cutoff() -> None:
    agg = aggregate([score(2), score(None)], k=5, cutoffs=[5])

    assert agg.mrr_at_k[5] == pytest.approx(0.25)


def test_a_hit_beyond_the_cutoff_contributes_nothing_to_mrr() -> None:
    agg = aggregate([score(9)], k=10, cutoffs=[5])

    assert agg.mrr_at_k[5] == 0.0


def test_aggregating_no_scores_yields_zeros_rather_than_dividing_by_zero() -> None:
    agg = aggregate([], k=5, cutoffs=[1, 5])

    assert agg.n_questions == 0
    assert agg.recall_at_k == {1: 0.0, 5: 0.0}
    assert agg.ndcg_at_k == 0.0


def test_as_dict_rounds_and_stringifies_cutoffs_for_json() -> None:
    payload = aggregate([score(1, ndcg=1 / 3)], k=5, cutoffs=[5]).as_dict()

    assert payload["recall_at_k"] == {"5": 1.0}
    assert payload["ndcg_at_k"] == 0.3333
