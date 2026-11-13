"""Eval-runner tests.

The runner's own arithmetic is thin — it delegates to `metrics`. What is worth
pinning is the bookkeeping that decides whether a number is allowed to be
believed: the `authoritative` flag that keeps unverified golden-set numbers out
of the README, and the config record that makes a result file reconstructible.
Both are process guarantees, and process guarantees are exactly what rots
silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.ingestion.pipeline import EmbeddedChunk
from app.retrieval.base import RetrievedChunk
from eval.harness.corpus_index import EvalCorpus
from eval.harness.golden_set import GoldenQuestion, SupportingSpan
from eval.harness.metrics import QuestionScore
from eval.harness.runner import (
    RunConfig,
    _breakdown,
    _score_question,
    format_summary,
    run,
    write_result,
)


def make_question(
    qid: str, span_text: str, verified: bool, qtype: str = "factual"
) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        question=f"question about {span_text}",
        answer="answer",
        type=qtype,  # type: ignore[arg-type]
        relevance="any",
        supporting_spans=(
            SupportingSpan(source_filename="paper.pdf", page_number=1, span=span_text),
        ),
        verified=verified,
    )


class StubCorpus(EvalCorpus):
    """An `EvalCorpus` whose retriever is scripted, so `run` needs no model."""

    def __init__(self, texts: list[str], ranked: list[str]):
        chunks = [
            EmbeddedChunk(
                vector_id=i,
                chunk_index=i,
                page_number=1,
                text=text,
                source_filename="paper.pdf",
            )
            for i, text in enumerate(texts)
        ]
        super().__init__(chunks, np.zeros((len(texts), 4), dtype="float32"), 1000, 150)
        # EvalCorpus is a frozen dataclass, so plain assignment would raise.
        object.__setattr__(self, "_ranked", ranked)

    def build_retriever(self, strategy, *, candidate_k=None, rrf_k=None, reranker=None):  # type: ignore[no-untyped-def]
        ranked = self._ranked

        class Scripted:
            def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
                return [
                    RetrievedChunk(
                        vector_id=i,
                        text=text,
                        page_number=1,
                        chunk_index=i,
                        source_filename="paper.pdf",
                        score=1.0 / (i + 1),
                    )
                    for i, text in enumerate(ranked[:k])
                ]

        return Scripted()


# --- config ----------------------------------------------------------------


def test_config_label_encodes_the_parameters_that_change_the_numbers() -> None:
    assert RunConfig(strategy="dense", chunk_size=500, chunk_overlap=50).label == "dense-cs500-co50"


# --- authoritative gating --------------------------------------------------


def test_a_run_over_an_unverified_golden_set_is_not_authoritative() -> None:
    """The guardrail from CLAUDE.md §8: never let an unverified number look final."""
    corpus = StubCorpus(["alpha text"], ["alpha text"])
    result = run(
        RunConfig(),
        questions=[make_question("q1", "alpha", verified=False)],
        corpus=corpus,
        require_verified=1.0,
    )

    assert result.verified_fraction == 0.0
    assert result.authoritative is False
    assert "NOT AUTHORITATIVE" in format_summary(result)


def test_a_fully_verified_run_meeting_the_threshold_is_authoritative() -> None:
    corpus = StubCorpus(["alpha text"], ["alpha text"])
    result = run(
        RunConfig(),
        questions=[make_question("q1", "alpha", verified=True)],
        corpus=corpus,
        require_verified=1.0,
    )

    assert result.authoritative is True
    assert "NOT AUTHORITATIVE" not in format_summary(result)


def test_the_default_threshold_of_zero_never_marks_a_run_authoritative() -> None:
    """Development runs must not be able to accidentally claim authority."""
    corpus = StubCorpus(["alpha text"], ["alpha text"])
    result = run(
        RunConfig(),
        questions=[make_question("q1", "alpha", verified=True)],
        corpus=corpus,
    )

    assert result.authoritative is False


# --- scoring ---------------------------------------------------------------


def test_run_scores_a_hit_and_a_miss_over_the_same_ranking() -> None:
    corpus = StubCorpus(["alpha text", "beta text"], ["alpha text", "beta text"])
    result = run(
        RunConfig(),
        questions=[
            make_question("hit", "beta", verified=False),
            make_question("miss", "gamma", verified=False),
        ],
        corpus=corpus,
        cutoffs=[1, 5],
    )

    assert result.scores.recall_at_k[5] == pytest.approx(0.5)
    assert result.scores.recall_at_k[1] == 0.0  # "beta" is at rank 2
    assert result.n_chunks == 2


def test_score_question_records_deduplicated_sources_in_rank_order() -> None:
    score = _score_question(
        make_question("q", "alpha", verified=False),
        ["alpha text", "more"],
        ["b.pdf", "a.pdf", "b.pdf"],
        ["alpha text", "more"],
        k=5,
    )

    assert score.retrieved_sources == ("b.pdf", "a.pdf")
    assert score.rank == 1


# --- breakdown -------------------------------------------------------------


def test_breakdown_reports_recall_per_question_type() -> None:
    """The per-type split is what supports the 'BM25 helps on numeric' claim."""
    scores = [
        QuestionScore("a", "numeric", 1, 1.0, 1, ()),
        QuestionScore("b", "numeric", None, 0.0, 1, ()),
        QuestionScore("c", "factual", 2, 0.6, 1, ()),
    ]

    rows = _breakdown(scores, k=5)

    assert rows["numeric"]["n"] == 2
    assert rows["numeric"]["recall_at_5"] == pytest.approx(0.5)
    assert rows["factual"]["recall_at_5"] == pytest.approx(1.0)


# --- persistence -----------------------------------------------------------


def test_write_result_round_trips_the_full_config(tmp_path: Path) -> None:
    """A result you cannot reconstruct the configuration of is not evidence."""
    corpus = StubCorpus(["alpha text"], ["alpha text"])
    # Chunking is pinned explicitly rather than left to default: this asserts
    # the filename encodes the config, not that the config is any one value.
    # Leaving it implicit made the test fail when Day 9's sweep moved the
    # default to 600/90 — a config change, not a regression.
    result = run(
        RunConfig(strategy="sparse", chunk_size=1000, chunk_overlap=150),
        questions=[make_question("q1", "alpha", False)],
        corpus=corpus,
    )

    path = write_result(result, tmp_path)
    payload = json.loads(path.read_text())

    assert path.name == "retrieval-sparse-cs1000-co150.json"
    assert payload["config"]["strategy"] == "sparse"
    assert payload["config"]["chunk_size"] == 1000
    assert payload["authoritative"] is False
    assert len(payload["per_question"]) == 1
