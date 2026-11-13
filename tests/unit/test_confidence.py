"""The abstention gate, and the misuse it has to refuse.

The gate's failure modes are all silent — a threshold pointed at the wrong
score doesn't raise, it just turns the abstention rate into a constant. So the
tests here are mostly about the boundary conditions and the guard.
"""

from __future__ import annotations

import pytest

from app.generation.schema import ModelAnswer, ModelClaim
from app.pipeline import QueryPipeline, UncalibratedGateError
from app.retrieval.base import RetrievedChunk
from app.retrieval.confidence import ConfidenceGate


def chunk(vector_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        vector_id=vector_id,
        text=f"chunk {vector_id}",
        page_number=1,
        chunk_index=vector_id,
        source_filename="paper.pdf",
        score=score,
    )


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self._chunks[:k]


class StubClient:
    def __init__(self, answer: ModelAnswer | None = None):
        self._answer = answer or ModelAnswer(
            sufficient_evidence=True,
            claims=[ModelClaim(text="An answer.", source_ids=[1])],
        )
        self.calls = 0

    def generate(self, prompt: str) -> ModelAnswer:
        self.calls += 1
        return self._answer


class CalibratedStub(StubRetriever):
    """A stub that truthfully declares its scores are cross-encoder logits."""

    emits_calibrated_scores = True


def reranking(chunks: list[RetrievedChunk]) -> CalibratedStub:
    return CalibratedStub(chunks)


# --- the gate itself -------------------------------------------------------


def test_no_chunks_abstains() -> None:
    decision = ConfidenceGate(threshold=0.0).evaluate([])
    assert decision.abstained and decision.top_score is None


def test_everything_below_the_threshold_abstains() -> None:
    decision = ConfidenceGate(threshold=0.0).evaluate([chunk(0, -8.2), chunk(1, -9.9)])
    assert decision.abstained
    assert decision.top_score == -8.2
    assert "-8.20" in decision.reason


def test_a_chunk_exactly_on_the_threshold_passes() -> None:
    """>=, not >. An off-by-one here silently shifts every calibrated number."""
    decision = ConfidenceGate(threshold=2.5).evaluate([chunk(0, 2.5)])
    assert decision.passed


def test_chunks_below_the_threshold_are_dropped_from_what_reaches_the_model() -> None:
    """A chunk scored -9 is not evidence; passing it invites the model to cite noise."""
    decision = ConfidenceGate(threshold=0.0).evaluate([chunk(0, 6.1), chunk(1, -7.0)])
    assert decision.passed
    assert [c.vector_id for c in decision.chunks] == [0]


def test_the_top_score_does_not_depend_on_caller_ordering() -> None:
    decision = ConfidenceGate(threshold=0.0).evaluate([chunk(0, 1.0), chunk(1, 9.0)])
    assert decision.top_score == 9.0


def test_min_supporting_chunks_abstains_on_a_single_weak_hit() -> None:
    gate = ConfidenceGate(threshold=0.0, min_supporting_chunks=2)
    decision = gate.evaluate([chunk(0, 6.0), chunk(1, -1.0)])
    assert decision.abstained
    assert "need 2" in decision.reason


# --- the guard -------------------------------------------------------------


def test_a_gate_behind_uncalibrated_scores_is_refused() -> None:
    """Cosine and RRF scores are not logits — this would silently stop gating."""
    with pytest.raises(UncalibratedGateError, match="cross-encoder logits"):
        QueryPipeline(StubRetriever([chunk(0, 0.9)]), gate=ConfidenceGate(threshold=0.0))  # type: ignore[arg-type]


def test_the_real_retrievers_declare_their_score_calibration_honestly() -> None:
    """The guard is only as good as these flags; a wrong one re-opens the hole."""
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.fusion import HybridRetriever
    from app.retrieval.reranker import RerankingRetriever
    from app.retrieval.sparse import SparseRetriever

    assert RerankingRetriever.emits_calibrated_scores is True
    assert DenseRetriever.emits_calibrated_scores is False
    assert SparseRetriever.emits_calibrated_scores is False
    assert HybridRetriever.emits_calibrated_scores is False


def test_an_ungated_pipeline_accepts_any_retriever() -> None:
    """The Day 11 ablation needs a no-gate arm to show what the gate buys."""
    pipeline = QueryPipeline(StubRetriever([chunk(0, 0.9)]), client=StubClient())  # type: ignore[arg-type]
    assert pipeline.answer("q").answered


# --- the pipeline ----------------------------------------------------------


def test_the_gate_abstains_without_spending_a_generation_call() -> None:
    """The whole point of gating before generation rather than after."""
    client = StubClient()
    pipeline = QueryPipeline(
        reranking([chunk(0, -8.0)]), gate=ConfidenceGate(threshold=0.0), client=client
    )
    result = pipeline.answer("q")

    assert not result.answered
    assert client.calls == 0


def test_gate_and_model_abstentions_stay_distinguishable() -> None:
    """Calibration needs to know which half refused, or it cannot tune either."""
    gate_result = QueryPipeline(
        reranking([chunk(0, -8.0)]), gate=ConfidenceGate(threshold=0.0), client=StubClient()
    ).answer("q")
    assert gate_result.abstain_source == "gate"

    model_result = QueryPipeline(
        reranking([chunk(0, 7.0)]),
        gate=ConfidenceGate(threshold=0.0),
        client=StubClient(ModelAnswer(sufficient_evidence=False, abstain_reason="nope")),
    ).answer("q")
    assert model_result.abstain_source == "model"


def test_an_answered_result_has_no_abstain_source() -> None:
    result = QueryPipeline(
        reranking([chunk(0, 7.0)]), gate=ConfidenceGate(threshold=0.0), client=StubClient()
    ).answer("q")
    assert result.answered and result.abstain_source is None


def test_the_result_keeps_every_retrieved_chunk_including_ones_the_gate_dropped() -> None:
    """The demo shows what the reranker saw, not only what survived the gate."""
    retrieved = [chunk(0, 7.0), chunk(1, -8.0)]
    answered = QueryPipeline(
        reranking(retrieved), gate=ConfidenceGate(threshold=0.0), client=StubClient()
    ).answer("q")
    refused = QueryPipeline(
        reranking([chunk(2, -8.0)]), gate=ConfidenceGate(threshold=0.0), client=StubClient()
    ).answer("q")

    assert answered.retrieved == tuple(retrieved)
    assert len(answered.gate.chunks) == 1
    assert [c.vector_id for c in refused.retrieved] == [2]


# --- calibration -----------------------------------------------------------


def test_the_sweep_uses_the_same_admits_call_the_service_uses() -> None:
    """A harness with its own copy of the gate logic calibrates the wrong gate."""
    from eval.harness.adversarial import AdversarialQuestion
    from eval.harness.calibrate import sweep

    adversarial = [
        AdversarialQuestion("a1", "q", "out_of_domain", "why", "gate"),
        AdversarialQuestion("a2", "q", "absent_detail", "why", "model"),
    ]
    rows = sweep(
        answerable_scores=[[5.0], [4.0]],
        adversarial_scores=[[-9.0], [6.0]],
        adversarial=adversarial,
        thresholds=[0.0],
    )
    row = rows[0]
    assert row.answerable_pass_rate == 1.0  # both answerable clear 0.0
    assert row.gate_catch_rate == 1.0  # the out_of_domain one is refused
    assert row.model_expected_caught_early == 0.0  # the absent_detail one passes
    assert row.youden_j == 1.0


def test_selection_will_not_trade_away_answerable_questions_for_catch_rate() -> None:
    """A gate that refuses real questions is a broken product, whatever it catches."""
    from eval.harness.calibrate import ThresholdRow, select

    greedy = ThresholdRow(8.0, 0.20, 1.00, 1.00, 1.00, 0.20)
    balanced = ThresholdRow(-2.0, 0.95, 0.93, 0.43, 0.15, 0.88)
    assert select([greedy, balanced], min_answerable_pass_rate=0.90) is balanced


def test_when_no_threshold_meets_the_floor_the_least_bad_one_is_reported() -> None:
    from eval.harness.calibrate import ThresholdRow, select

    a = ThresholdRow(4.0, 0.60, 1.0, 1.0, 1.0, 0.60)
    b = ThresholdRow(8.0, 0.20, 1.0, 1.0, 1.0, 0.20)
    assert select([a, b], min_answerable_pass_rate=0.95) is a
