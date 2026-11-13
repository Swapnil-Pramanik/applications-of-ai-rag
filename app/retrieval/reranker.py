"""Cross-encoder reranking over a fused candidate pool.

**Why this stage exists at all.** Everything upstream scores a query and a
passage *separately* — the dense retriever compares two independently computed
384-dim vectors, BM25 compares term statistics, and RRF only compares ranks.
None of them ever look at the query and the passage together. A cross-encoder
does: it concatenates them into one input and runs full attention across the
pair, so it can tell that a passage mentioning "batch size 32" is the answer to
"what batch size did they train with" rather than merely nearby in topic space.

That is strictly more accurate and strictly more expensive. A bi-encoder embeds
1,446 chunks once, at ingestion, and every later query is a dot product; a
cross-encoder has to run the model once per (query, candidate) pair, at query
time, and cannot be precomputed. Hence the shape of this pipeline: retrieve
wide and cheap (`candidate_k=20` per retriever), rerank narrow and expensive
(20 pairs), return `top_k=5`. Reranking the whole corpus would be ~1,400
forward passes per query and is not on the table.

**The specific problem it is meant to solve here.** The Day 6-7 ablation found
plain BM25 (Recall@5 0.550) beating RRF hybrid (0.483): fusing on ranks means a
mediocre dense rank-1 hit carries the same weight as a confident BM25 rank-1
hit, so dense noise displaces correct BM25 results out of the top 5. A
cross-encoder rescores the fused pool on true query-document relevance, which
should let the pool stay wide (RRF's recall advantage at k=20 is real —
hybrid Recall@10 is 0.583) while fixing the ordering that RRF got wrong. If it
doesn't, the honest fallback is weighted fusion or BM25-primary.

**`ms-marco-MiniLM-L-6-v2`** is 6 layers / ~22M params, trained on MS MARCO
passage ranking. Chosen because it is the smallest widely-benchmarked
cross-encoder that still clearly beats bi-encoder ordering, it runs on CPU in
tens of milliseconds for 20 pairs, and it is free — the same $0 / 16GB
constraints that picked `bge-small`. Its scores are raw logits, unbounded and
uncalibrated, so they are meaningful for *ordering* only. The Day 11 abstention
gate needs a probability, not a logit, which is why it will calibrate a
threshold against these scores rather than assume one.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import settings
from app.retrieval.base import RetrievedChunk, Retriever

if TYPE_CHECKING:  # pragma: no cover - deferred so importing this is cheap
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    # Annotated because `CrossEncoder.__init__` is untyped upstream, so mypy
    # infers Any and then objects to returning it.
    model: CrossEncoder = CrossEncoder(model_name)
    return model


class Reranker:
    """Scores (query, passage) pairs jointly. `model` is injectable for tests."""

    def __init__(self, model: CrossEncoder | None = None, model_name: str | None = None):
        self._model = model
        self._model_name = model_name or settings.reranker_model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = _load_model(self._model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance logits, one per text, in the order given."""
        if not texts:
            return []
        return [float(score) for score in self.model.predict([(query, text) for text in texts])]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


class RerankingRetriever:
    """Wraps any `Retriever`: fetch `candidate_k`, rescore, return the best `k`.

    Composes rather than subclasses so the ablation can wrap *any* base — the
    point of Day 8 is comparing `hybrid` against `hybrid+rerank` and
    `sparse+rerank` over the identical candidate pool, which is only a fair
    comparison if the base retriever is untouched.

    The returned `score` is the cross-encoder logit, replacing whatever the
    base retriever produced. That is a deliberate break in meaning: after this
    stage, `score` is the only number in the pipeline that is comparable across
    queries at all, which is what makes a calibrated abstention threshold
    possible in the first place.
    """

    # The cross-encoder logit replaces the base retriever's score, and it is
    # the one number in this pipeline comparable across queries — which is what
    # lets the Day 11 abstention gate sit behind this stage and nowhere else.
    emits_calibrated_scores = True

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker | None = None,
        candidate_k: int | None = None,
    ):
        self._base = base
        self._reranker = reranker or get_reranker()
        self._candidate_k = candidate_k if candidate_k is not None else settings.candidate_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []

        # Always pull the full candidate pool, even when k is small: reranking
        # only helps if it is given something to reorder. Asking the base for
        # k would make this stage a no-op.
        candidates = self._base.retrieve(query, max(self._candidate_k, k))
        if not candidates:
            return []

        scores = self._reranker.score(query, [chunk.text for chunk in candidates])
        # Ties broken by vector_id so the ranking is deterministic run to run —
        # an eval harness that reorders ties silently perturbs Recall@k.
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].vector_id),
        )
        return [replace(chunk, score=score) for chunk, score in ranked[:k]]
