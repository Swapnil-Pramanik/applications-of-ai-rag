"""The demo's decision logic, tested without Streamlit, models or Gemini.

Uses the BM25 arms so no embedding model is needed; the cross-encoder and the
LLM are stubs that go in through the same parameters production uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.generation.gemini import GenerationUnavailableError
from app.generation.schema import Answer, Citation, Claim, ModelAnswer, ModelClaim
from app.ingestion.embedder import Embedder
from app.ingestion.pipeline import EmbeddedChunk
from app.retrieval.reranker import Reranker
from eval.harness.corpus_index import STRATEGIES, EvalCorpus
from ui.service import (
    EXAMPLES,
    STRATEGY_LABELS,
    DemoService,
    NoSourcesError,
    UploadError,
    numbered_claims,
)

TEXTS = [
    ("1412.6980.pdf", "adam default settings are alpha 0.001 beta1 0.9 beta2 0.999"),
    ("1706.03762.pdf", "we used the adam optimizer with beta2 0.98"),
    ("1512.03385.pdf", "residual learning eases the training of deep networks"),
]


def make_corpus() -> EvalCorpus:
    chunks = [
        EmbeddedChunk(vector_id=i, chunk_index=i, page_number=i + 1, text=t, source_filename=f)
        for i, (f, t) in enumerate(TEXTS)
    ]
    vectors = np.ones((len(chunks), settings.embedding_dim), dtype="float32")
    return EvalCorpus(chunks, vectors, 600, 90)


class FixedReranker(Reranker):
    """Every passage gets the same cross-encoder score."""

    def __init__(self, score: float):
        super().__init__(model=object())
        self._score = score

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self._score] * len(texts)


class StubClient:
    def __init__(self, answer: ModelAnswer | None = None, error: Exception | None = None):
        self._answer = answer or ModelAnswer(
            sufficient_evidence=True, claims=[ModelClaim(text="Beta2 is 0.999.", source_ids=[1])]
        )
        self._error = error
        self.calls = 0

    def generate(self, prompt: str) -> ModelAnswer:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._answer


def service(score: float = 5.0, client: StubClient | None = None) -> DemoService:
    return DemoService(make_corpus(), reranker=FixedReranker(score), client=client or StubClient())


QUESTION = "adam beta2 default"


def test_a_confident_retrieval_is_answered_with_citations() -> None:
    result = service().ask(QUESTION, "sparse+rerank")

    assert result.status == "answered"
    assert result.answer is not None and result.answer.citations
    assert len(result.retrieved) > 0


def test_the_gate_refuses_without_calling_the_llm() -> None:
    client = StubClient()
    result = service(score=-9.0, client=client).ask(QUESTION, "sparse+rerank")

    assert result.status == "refused_by_gate"
    assert client.calls == 0
    # The audience still sees what was retrieved, and why it was not enough.
    assert result.retrieved and all(result.passed_gate(c) is False for c in result.retrieved)


def test_a_model_refusal_is_reported_as_the_models() -> None:
    client = StubClient(ModelAnswer(sufficient_evidence=False, abstain_reason="not stated"))
    result = service(client=client).ask(QUESTION, "sparse+rerank")

    assert result.status == "refused_by_model"


def test_retrieval_only_mode_spends_no_llm_call_but_still_runs_the_gate() -> None:
    client = StubClient()
    result = service(client=client).ask(QUESTION, "sparse+rerank", generate=False)

    assert result.status == "retrieval_only"
    assert client.calls == 0
    assert result.gate is not None and result.gate.passed


def test_uncalibrated_retrievers_run_without_a_gate() -> None:
    """A -2.0 logit threshold means nothing against BM25 scores, so it is off."""
    result = service().ask(QUESTION, "sparse")

    assert result.gate is None
    assert result.passed_gate(result.retrieved[0]) is None
    assert result.status == "answered"


def test_a_quota_error_is_an_error_not_a_refusal_and_keeps_retrieval() -> None:
    client = StubClient(error=GenerationUnavailableError("quota", status_code=429))
    result = service(client=client).ask(QUESTION, "sparse+rerank")

    assert result.status == "error"
    assert result.error is not None and "quota" in result.error
    assert result.retrieved


def test_retrievers_are_built_once_per_strategy() -> None:
    svc = service()
    assert svc.retriever("sparse") is svc.retriever("sparse")


def test_papers_are_listed_by_title_with_chunk_counts() -> None:
    papers = dict(service().papers())

    assert papers["Adam Optimizer"] == 1
    assert service().chunk_count == 3


def test_claims_sharing_a_source_share_its_footnote_number() -> None:
    a = Citation("a.pdf", 1, 0, 0, "q")
    b = Citation("b.pdf", 2, 1, 1, "q")
    answer = Answer(
        answered=True,
        claims=(Claim("first", (a,)), Claim("second", (b, a))),
    )

    assert [numbers for _, numbers in numbered_claims(answer)] == [[1], [1, 2]]


def test_every_strategy_has_a_label_and_examples_are_non_empty() -> None:
    assert set(STRATEGY_LABELS) == set(STRATEGIES)
    assert all(example.question for example in EXAMPLES)


class OnesModel:
    """Embedding stub: every text maps to the same unit-ish vector."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.ones((len(texts), settings.embedding_dim), dtype=np.float32)


def uploading_service() -> DemoService:
    return DemoService(
        make_corpus(),
        reranker=FixedReranker(5.0),
        client=StubClient(),
        embedder=Embedder(model=OnesModel()),  # type: ignore[arg-type]
    )


def test_an_uploaded_document_becomes_searchable(tmp_path: Path) -> None:
    demo = uploading_service()
    before = demo.chunk_count
    demo.retriever("sparse")  # built before the upload; must not be reused after it
    doc = tmp_path / "notes.txt"
    doc.write_text("the zebrafish optimizer uses a learning rate of 0.42", encoding="utf-8")

    result = demo.add_document(doc)

    assert result.filename == "notes.txt" and result.chunks >= 1
    assert demo.chunk_count == before + result.chunks
    assert demo.uploaded == ["notes.txt"]
    top = demo.ask("zebrafish optimizer", "sparse", generate=False).retrieved[0]
    assert top.source_filename == "notes.txt"
    # Vector ids continue after the corpus rather than colliding with it.
    assert top.vector_id >= len(TEXTS)


def test_uploading_a_duplicate_or_empty_file_is_rejected(tmp_path: Path) -> None:
    demo = uploading_service()
    duplicate = tmp_path / "1412.6980.pdf"
    duplicate.write_bytes(b"irrelevant")
    with pytest.raises(UploadError, match="already in the corpus"):
        demo.add_document(duplicate)

    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    with pytest.raises(UploadError, match="no text"):
        demo.add_document(empty)
    assert demo.uploaded == []


def test_questions_search_only_the_selected_sources() -> None:
    # Three chunks per file, "optimizer" in one chunk of each: enough chunks for BM25's IDF.
    texts = [
        (f, t)
        for f in ("a.txt", "b.txt")
        for t in ("the optimizer setting", "residual blocks", "image patches")
    ]
    chunks = [
        EmbeddedChunk(vector_id=i, chunk_index=i, page_number=1, text=t, source_filename=f)
        for i, (f, t) in enumerate(texts)
    ]
    corpus = EvalCorpus(chunks, np.ones((6, settings.embedding_dim), dtype="float32"), 600, 90)
    demo = DemoService(corpus, reranker=FixedReranker(5.0), client=StubClient())

    only_b = demo.ask("optimizer", "sparse", generate=False, sources=["b.txt"])
    assert [c.source_filename for c in only_b.retrieved] == ["b.txt"]
    both = demo.ask("optimizer", "sparse", generate=False)
    assert {c.source_filename for c in both.retrieved} == {"a.txt", "b.txt"}


def test_asking_with_no_matching_source_is_an_error() -> None:
    with pytest.raises(NoSourcesError):
        service().ask("adam", "sparse", generate=False, sources=["missing.pdf"])
    with pytest.raises(NoSourcesError):
        DemoService(EvalCorpus.empty(), reranker=FixedReranker(5.0)).ask(
            "adam", "sparse", generate=False
        )


def test_a_removed_source_is_no_longer_searched(tmp_path: Path) -> None:
    demo = uploading_service()
    doc = tmp_path / "notes.txt"
    doc.write_text("zebrafish optimizer", encoding="utf-8")
    demo.add_document(doc)
    demo.remove_document("notes.txt")
    assert demo.uploaded == [] and "notes.txt" not in demo.chunk_counts()
    top = demo.ask("zebrafish optimizer", "sparse", generate=False).retrieved
    assert all(c.source_filename != "notes.txt" for c in top)
    with pytest.raises(KeyError):
        demo.remove_document("notes.txt")
