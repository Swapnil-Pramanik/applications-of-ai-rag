"""Everything the demo page needs, with no Streamlit in it.

The page (`ui/streamlit_app.py`) only draws; this module decides. Keeping them
apart means the logic is unit-tested like the rest of the project, and the page can be
redesigned later without touching anything that affects an answer.

The demo answers through the same `QueryPipeline` the eval harness measures, so
what the audience sees is the system the numbers describe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from app.core.config import settings
from app.core.corpus import title_for
from app.generation.gemini import GenerationUnavailableError, StructuredClient
from app.generation.schema import Answer, Citation, Claim
from app.pipeline import QueryPipeline
from app.retrieval.base import RetrievedChunk, Retriever
from app.retrieval.confidence import ConfidenceGate, GateDecision
from app.retrieval.reranker import Reranker
from eval.harness.corpus_index import STRATEGIES, EvalCorpus, load_corpus

# Shown in the retriever dropdown, best first. Keys are the eval harness's
# strategy names, so the demo and the ablation table use the same arms.
STRATEGY_LABELS: dict[str, str] = {
    "hybrid+rerank": "Hybrid + rerank (default)",
    "sparse+rerank": "BM25 + rerank",
    "dense+rerank": "Dense + rerank",
    "hybrid": "Hybrid (RRF), no rerank",
    "sparse": "BM25 only",
    "dense": "Dense only",
}
assert set(STRATEGY_LABELS) == set(STRATEGIES)

Status = Literal["answered", "refused_by_gate", "refused_by_model", "retrieval_only", "error"]


@dataclass(frozen=True)
class Example:
    label: str
    question: str


# Picked to show each behaviour once in a live demo: a plain fact, a
# comparison, the multi-hop weakness, and both kinds of refusal. The first two
# are golden questions g002 and g041, which hybrid+rerank retrieves at rank 1
# (eval/results/retrieval-hybrid-rerank-cs600-co90.json), so they answer
# reliably. The multi-hop one is g049, which retrieval misses: the model
# should refuse rather than fill in Adam's default from memory.
EXAMPLES: tuple[Example, ...] = (
    Example(
        "Fact",
        "How many attention heads does the base Transformer use, and what is the dimension "
        "of each head?",
    ),
    Example(
        "Comparison",
        "Which two earlier optimizers is Adam designed to combine, and what does it take "
        "from them?",
    ),
    Example(
        "Multi-hop",
        "Which optimizer did the Transformer authors use, and how does their β2 differ "
        "from that optimizer's own default?",
    ),
    Example("Off-topic", "What is the capital of France?"),
    Example(
        "Not in paper", "What was the total dollar cost of training the Transformer base model?"
    ),
)


@dataclass(frozen=True)
class DemoResult:
    question: str
    strategy: str
    retrieved: tuple[RetrievedChunk, ...]
    # None when the retriever's scores are not cross-encoder logits: the gate's
    # threshold would be meaningless against them, so it is switched off.
    gate: GateDecision | None
    # None when generation was switched off, or when Gemini was unavailable.
    answer: Answer | None
    error: str | None
    seconds: float

    @property
    def status(self) -> Status:
        if self.error is not None:
            return "error"
        if self.gate is not None and self.gate.abstained:
            return "refused_by_gate"
        if self.answer is None:
            return "retrieval_only"
        return "answered" if self.answer.answered else "refused_by_model"

    def passed_gate(self, chunk: RetrievedChunk) -> bool | None:
        """Whether this chunk cleared the threshold; None when there is no gate."""
        if self.gate is None:
            return None
        return chunk.score >= self.gate.threshold


def numbered_claims(answer: Answer) -> list[tuple[Claim, list[int]]]:
    """Each claim with the 1-based numbers of its sources in `answer.citations`.

    `answer.citations` is de-duplicated, so two claims citing the same chunk
    share a number — which is what a reader expects from footnotes.
    """
    position = {_key(c): i for i, c in enumerate(answer.citations, start=1)}
    return [
        (claim, sorted({position[_key(c)] for c in claim.citations})) for claim in answer.claims
    ]


def _key(citation: Citation) -> tuple[str, int, int]:
    return (citation.source_filename, citation.page_number, citation.chunk_index)


def paper_title(source_filename: str) -> str:
    return title_for(source_filename)


def explain_outage(error: GenerationUnavailableError) -> str:
    if error.status_code == 429:
        return "Gemini's free-tier quota is used up for now. Showing retrieval only."
    if error.status_code == 503:
        return (
            "Gemini is overloaded right now (503). Try again in a minute. Showing retrieval only."
        )
    return f"Gemini could not be reached ({error}). Showing retrieval only."


class DemoService:
    def __init__(
        self,
        corpus: EvalCorpus,
        *,
        reranker: Reranker | None = None,
        client: StructuredClient | None = None,
        top_k: int | None = None,
    ):
        self._corpus = corpus
        # One cross-encoder shared by every +rerank arm, so switching retrievers
        # in the dropdown does not load the model again.
        self._reranker = reranker or Reranker()
        self._client = client
        self._top_k = top_k if top_k is not None else settings.top_k
        self._retrievers: dict[str, Retriever] = {}

    @property
    def chunk_count(self) -> int:
        return len(self._corpus.chunks)

    def papers(self) -> list[tuple[str, int]]:
        """(title, number of chunks) for every paper, alphabetically."""
        counts: dict[str, int] = {}
        for chunk in self._corpus.chunks:
            title = paper_title(chunk.source_filename)
            counts[title] = counts.get(title, 0) + 1
        return sorted(counts.items())

    def retriever(self, strategy: str) -> Retriever:
        if strategy not in self._retrievers:
            self._retrievers[strategy] = self._corpus.build_retriever(
                strategy, reranker=self._reranker
            )
        return self._retrievers[strategy]

    def ask(self, question: str, strategy: str, *, generate: bool = True) -> DemoResult:
        started = time.perf_counter()
        retriever = self.retriever(strategy)
        gate = ConfidenceGate() if getattr(retriever, "emits_calibrated_scores", False) else None

        if generate:
            pipeline = QueryPipeline(retriever, gate=gate, client=self._client, top_k=self._top_k)
            try:
                result = pipeline.answer(question)
            except GenerationUnavailableError as error:
                # An outage is not a refusal. Say so, and still show retrieval,
                # so a quota error mid-presentation leaves something to talk about.
                return self._retrieval_only(
                    question, strategy, retriever, gate, started, error=explain_outage(error)
                )
            return DemoResult(
                question=question,
                strategy=strategy,
                retrieved=result.retrieved,
                gate=result.gate if gate is not None else None,
                answer=result.answer,
                error=None,
                seconds=time.perf_counter() - started,
            )

        return self._retrieval_only(question, strategy, retriever, gate, started)

    def _retrieval_only(
        self,
        question: str,
        strategy: str,
        retriever: Retriever,
        gate: ConfidenceGate | None,
        started: float,
        *,
        error: str | None = None,
    ) -> DemoResult:
        chunks = retriever.retrieve(question, self._top_k)
        return DemoResult(
            question=question,
            strategy=strategy,
            retrieved=tuple(chunks),
            gate=gate.evaluate(chunks) if gate is not None else None,
            answer=None,
            error=error,
            seconds=time.perf_counter() - started,
        )


def load_service() -> DemoService:
    """Ingest (or read the cached) corpus and build the service. Slow the first time."""
    return DemoService(load_corpus())
