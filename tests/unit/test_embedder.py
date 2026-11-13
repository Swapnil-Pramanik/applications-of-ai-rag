"""Embedder tests run against a stub model, so they stay offline and fast.

What's actually worth asserting here isn't "does sentence-transformers work"
(it does) but the two things this project gets wrong silently if they're
wrong: the query-only instruction prefix, and that normalization is requested
so FAISS inner product means cosine.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ingestion.embedder import QUERY_INSTRUCTION, Embedder


class StubModel:
    """Records what it was asked to encode and returns deterministic vectors."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.seen_texts: list[str] = []
        self.seen_kwargs: dict[str, object] = {}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.seen_texts = list(texts)
        self.seen_kwargs = kwargs
        return np.arange(len(texts) * self.dim, dtype=np.float32).reshape(len(texts), self.dim)


@pytest.fixture
def stub() -> StubModel:
    return StubModel()


def test_documents_are_embedded_without_an_instruction_prefix(stub: StubModel) -> None:
    Embedder(model=stub).embed_documents(["attention is all you need"])

    assert stub.seen_texts == ["attention is all you need"]


def test_queries_are_embedded_with_the_bge_instruction_prefix(stub: StubModel) -> None:
    Embedder(model=stub).embed_query("what is multi-head attention?")

    assert stub.seen_texts == [QUERY_INSTRUCTION + "what is multi-head attention?"]


def test_normalization_is_requested_so_inner_product_equals_cosine(stub: StubModel) -> None:
    Embedder(model=stub).embed_documents(["a"])

    assert stub.seen_kwargs["normalize_embeddings"] is True


def test_embed_query_returns_a_single_flat_vector(stub: StubModel) -> None:
    vector = Embedder(model=stub).embed_query("q")

    assert vector.shape == (stub.dim,)
    assert vector.dtype == np.float32


def test_embed_documents_returns_one_row_per_text(stub: StubModel) -> None:
    vectors = Embedder(model=stub).embed_documents(["a", "b", "c"])

    assert vectors.shape == (3, stub.dim)
    assert vectors.dtype == np.float32


def test_empty_input_short_circuits_without_touching_the_model(stub: StubModel) -> None:
    """An empty document (or a page of whitespace) must not crash ingestion."""
    vectors = Embedder(model=stub).embed_documents([])

    assert vectors.shape[0] == 0
    assert stub.seen_texts == []
