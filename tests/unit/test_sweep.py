"""The sweep's selection logic, tested without ingesting anything.

The expensive part of a sweep is ingestion, and it is already covered by
`test_corpus_index`. What is worth testing here is the part that decides
*which arm wins* — a wrong tie-break or a wrong overlap arithmetic silently
ships the wrong chunking, and nothing downstream would flag it.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.harness.corpus_index import EvalCorpus
from eval.harness.metrics import AggregateScores
from eval.harness.runner import RunConfig, RunResult
from eval.harness.sweep import BASE_OVERLAP_PCT, SIZE_ARMS, _sort_key, format_report, overlap_for


def make_result(chunk_size: int, chunk_overlap: int, recall_at_5: float, ndcg: float) -> RunResult:
    return RunResult(
        config=RunConfig(
            strategy="hybrid+rerank", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ),
        scores=AggregateScores(
            n_questions=60,
            k=10,
            recall_at_k={1: 0.3, 3: 0.4, 5: recall_at_5, 10: 0.65},
            mrr_at_k={1: 0.3, 3: 0.35, 5: 0.4, 10: 0.42},
            ndcg_at_k=ndcg,
        ),
        per_question=[],
        n_chunks=2277,
        verified_fraction=1.0,
        authoritative=True,
        by_type={
            "comparative": {"n": 12.0, "recall_at_5": 0.58, "ndcg": 0.56},
            "factual": {"n": 17.0, "recall_at_5": 0.65, "ndcg": 0.57},
            "multi_hop": {"n": 12.0, "recall_at_5": 0.17, "ndcg": 0.35},
            "numeric": {"n": 19.0, "recall_at_5": 0.74, "ndcg": 0.62},
        },
        timestamp="2026-08-27T00:00:00+00:00",
        platform="test",
    )


@pytest.mark.parametrize(
    ("chunk_size", "expected"), [(400, 60), (600, 90), (1000, 150), (1500, 225)]
)
def test_overlap_is_a_percentage_of_size_not_a_constant(chunk_size: int, expected: int) -> None:
    """A fixed character overlap would confound the very variable phase 1 isolates."""
    assert overlap_for(chunk_size, BASE_OVERLAP_PCT) == expected


def test_recall_at_5_decides_before_ndcg() -> None:
    worse_recall_better_ndcg = make_result(400, 60, recall_at_5=0.50, ndcg=0.90)
    better_recall = make_result(600, 90, recall_at_5=0.57, ndcg=0.10)
    assert max([worse_recall_better_ndcg, better_recall], key=_sort_key) is better_recall


def test_ndcg_breaks_a_recall_tie() -> None:
    """This is the tie that actually happened on Day 9: 600/90 and 1000/150 both hit 0.567."""
    cs600 = make_result(600, 90, recall_at_5=0.567, ndcg=0.539)
    cs1000 = make_result(1000, 150, recall_at_5=0.567, ndcg=0.487)
    assert max([cs1000, cs600], key=_sort_key) is cs600


def test_report_marks_the_winner_and_keeps_every_losing_arm() -> None:
    """CLAUDE.md's Day 9 line: record the losers too, or the pick is indistinguishable
    from a guess."""
    phase1 = [
        make_result(size, overlap_for(size, BASE_OVERLAP_PCT), 0.4, 0.4) for size in SIZE_ARMS
    ]
    winner = make_result(600, 90, 0.567, 0.539)
    phase2 = [make_result(600, 0, 0.417, 0.437), make_result(600, 180, 0.517, 0.499), winner]

    report = format_report(phase1, phase2, winner, "hybrid+rerank")

    for size in SIZE_ARMS:
        assert f"| {size} |" in report
    assert "| 0 (0%) |" in report and "| 180 (30%) |" in report
    # Twice, not once: the winning arm is the 15% point on the overlap curve as
    # well as a point on the size curve, so it is deliberately listed in both
    # tables. Dropping it from phase 2 would leave a hole where the incumbent sits.
    assert report.count("**<-- winner**") == 2
    assert "chunk_size=600, chunk_overlap=90" in report
    # per-type must survive into the report: the aggregate hides multi-hop
    assert "multi_hop 0.170" in report


def test_eval_corpus_exposes_the_chunking_it_was_built_at() -> None:
    """A result whose chunking you cannot recover is not evidence."""
    corpus = EvalCorpus([], np.empty((0, 4), dtype="float32"), 600, 90)
    assert (corpus.chunk_size, corpus.chunk_overlap) == (600, 90)
