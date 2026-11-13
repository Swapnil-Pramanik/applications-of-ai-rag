"""The abstention gate: refuse before generating, when retrieval found nothing.

**Why gate on cross-encoder logits and nothing else.** Every other score in
this pipeline is incomparable across queries. Cosine similarity from
`bge-small` drifts with query length and phrasing; BM25 sums unbounded term
weights, so its scale depends on corpus statistics and query length; RRF
returns a sum of reciprocal ranks, which by construction says nothing about
relevance — the top result of a catastrophic search and the top result of a
perfect one both score ~1/61. A threshold on any of those would mean something
different for every question asked.

The cross-encoder is the only stage that reads query and passage together, and
its logit is the one number in this system that is roughly comparable between
queries. That is what makes a fixed threshold meaningful at all — see
`reranker.RerankingRetriever`, which deliberately replaces `score` with it.

**This is the retrieval half of a two-part policy.** It catches "retrieval
returned nothing plausible" and costs no Gemini call. The generation half
(`generation/schema.ModelAnswer.sufficient_evidence`) catches the different
failure where plausible chunks came back but do not actually answer the
question. Neither subsumes the other.

**The threshold is calibrated, not guessed** — see `eval/harness/calibrate.py`,
which sweeps it against answerable and unanswerable questions and reports the
trade. Until that runs against a human-verified adversarial set, the default
here is an explicitly provisional prior and is labelled as one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import settings
from app.retrieval.base import RetrievedChunk


@dataclass(frozen=True)
class GateDecision:
    """Why the gate did what it did. Kept whole so telemetry can log the reason."""

    passed: bool
    chunks: tuple[RetrievedChunk, ...]
    top_score: float | None
    threshold: float
    reason: str

    @property
    def abstained(self) -> bool:
        return not self.passed


class ConfidenceGate:
    """Decides whether retrieved chunks are worth spending a generation call on.

    Two knobs, both doing different jobs:

    - `threshold` is the bar the *best* chunk must clear. Below it, nothing
      retrieved is plausibly about the question and we abstain outright.
    - `min_supporting_chunks` requires that many chunks above the bar before
      answering. Left at 1 by default: raising it is a lever for `relevance:
      "all"` questions, which need several pieces of evidence, but it also
      abstains on perfectly good single-source questions, so it is a calibration
      decision rather than an obvious win.

    Chunks below the threshold are dropped from what reaches the generator. A
    chunk the cross-encoder scored at -9 is not evidence; passing it along
    spends prompt budget and invites the model to cite noise.
    """

    def __init__(
        self,
        threshold: float | None = None,
        *,
        min_supporting_chunks: int | None = None,
    ):
        self.threshold = threshold if threshold is not None else settings.confidence_threshold
        self.min_supporting_chunks = (
            min_supporting_chunks
            if min_supporting_chunks is not None
            else settings.min_supporting_chunks
        )

    def admits(self, scores: Sequence[float]) -> bool:
        """The decision, over scores alone.

        Split out so `eval/harness/calibrate.py` can sweep thresholds against
        cached retrieval scores without re-running retrieval — and, more
        importantly, without reimplementing this rule. A calibration harness
        that carries its own copy of the gate logic eventually calibrates a
        gate the service does not run.
        """
        return sum(1 for score in scores if score >= self.threshold) >= self.min_supporting_chunks

    def evaluate(self, chunks: list[RetrievedChunk]) -> GateDecision:
        if not chunks:
            return GateDecision(
                passed=False,
                chunks=(),
                top_score=None,
                threshold=self.threshold,
                reason="retrieval returned no chunks",
            )

        # `RerankingRetriever` already returns these best-first, but the gate
        # must not depend on a caller's ordering to decide what "top" means.
        top_score = max(chunk.score for chunk in chunks)
        kept = tuple(chunk for chunk in chunks if chunk.score >= self.threshold)

        if not self.admits([chunk.score for chunk in chunks]):
            return GateDecision(
                passed=False,
                chunks=(),
                top_score=top_score,
                threshold=self.threshold,
                reason=(
                    f"only {len(kept)} chunk(s) scored at or above {self.threshold:.2f} "
                    f"(best was {top_score:.2f}); need {self.min_supporting_chunks}"
                ),
            )

        return GateDecision(
            passed=True,
            chunks=kept,
            top_score=top_score,
            threshold=self.threshold,
            reason=f"{len(kept)} chunk(s) at or above {self.threshold:.2f}",
        )
