"""One query, end to end: retrieve, gate, generate.

This is the seam the API layer calls and the eval harness measures, so the two
cannot drift — the same code path produces the demo's answer and the committed
abstention numbers.

**The gate is only meaningful behind calibrated scores, and that is enforced
here.** `ConfidenceGate` compares against a cross-encoder logit. Point it at a
bare dense retriever and it compares that same number against a cosine
similarity in [-1, 1] — every query clears a threshold of -2.0, or none does,
depending on the sign, and the gate silently stops gating. Nothing would raise;
the abstention rate would just quietly become a constant. So construction
rejects the combination rather than trusting the caller to remember.

The check is `retriever.emits_calibrated_scores`, not `isinstance(...,
RerankingRetriever)`. What matters is whether the numbers mean the same thing
between queries, which is a property of the scores — tying it to a class name
would reject a future calibrated retriever and accept a subclass that broke the
guarantee.

**Abstention has two causes and they stay distinguishable.** `gate` means
retrieval found nothing plausible and no Gemini call was made; `model` means
the model read plausible chunks and judged them insufficient. Day 11's
calibration needs to tell those apart to know which half to tune, and an
`AnswerResult` that collapsed them would make that impossible.

An upstream outage is neither — see `generation.gemini.GenerationUnavailableError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import settings
from app.generation.gemini import StructuredClient, generate_answer
from app.generation.schema import Answer
from app.retrieval.base import RetrievedChunk, Retriever
from app.retrieval.confidence import ConfidenceGate, GateDecision

AbstainSource = Literal["gate", "model", None]


@dataclass(frozen=True)
class AnswerResult:
    answer: Answer
    gate: GateDecision
    retrieved_count: int
    # Everything the retriever returned, best first — including chunks the gate
    # dropped. The demo UI shows these so a viewer can see *why* the gate
    # refused or what the model was given.
    retrieved: tuple[RetrievedChunk, ...] = ()

    @property
    def answered(self) -> bool:
        return self.answer.answered

    @property
    def abstain_source(self) -> AbstainSource:
        """Which half of the policy refused — the whole point of keeping both."""
        if self.answer.answered:
            return None
        return "gate" if self.gate.abstained else "model"


class UncalibratedGateError(ValueError):
    """Raised when a confidence gate is placed behind scores it cannot read."""


class QueryPipeline:
    def __init__(
        self,
        retriever: Retriever,
        *,
        gate: ConfidenceGate | None = None,
        client: StructuredClient | None = None,
        top_k: int | None = None,
    ):
        if gate is not None and not getattr(retriever, "emits_calibrated_scores", False):
            raise UncalibratedGateError(
                f"{type(retriever).__name__} does not emit calibrated scores, and "
                "ConfidenceGate thresholds are cross-encoder logits — meaningless against "
                "cosine, BM25 or RRF scores. Wrap it in a RerankingRetriever, or pass gate=None."
            )
        self._retriever = retriever
        self._gate = gate
        self._client = client
        self._top_k = top_k if top_k is not None else settings.top_k

    def answer(self, question: str) -> AnswerResult:
        chunks = self._retriever.retrieve(question, self._top_k)

        if self._gate is None:
            # Ungated is a legitimate configuration: the Day 11 ablation needs a
            # no-gate arm to show what the gate is actually buying.
            decision = GateDecision(
                passed=True,
                chunks=tuple(chunks),
                top_score=max((c.score for c in chunks), default=None),
                threshold=float("-inf"),
                reason="no gate configured",
            )
        else:
            decision = self._gate.evaluate(chunks)

        if not decision.passed:
            return AnswerResult(
                answer=Answer(answered=False, abstain_reason=decision.reason),
                gate=decision,
                retrieved_count=len(chunks),
                retrieved=tuple(chunks),
            )

        answer = generate_answer(question, list(decision.chunks), client=self._client)
        return AnswerResult(
            answer=answer, gate=decision, retrieved_count=len(chunks), retrieved=tuple(chunks)
        )
