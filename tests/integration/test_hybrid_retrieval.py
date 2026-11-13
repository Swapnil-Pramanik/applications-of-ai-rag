"""Hybrid retrieval against the real bge-small-en-v1.5 model.

The unit tests prove each retriever's contract in isolation. This one exists
to check the *premise* of the whole architecture on real embeddings: that
dense and sparse fail on different queries, and that fusion recovers both.
These are illustrative, not the measured result — the real numbers come from
the golden set on Day 6-9.

Downloads ~130MB on first run, so it is opt-in:

    RUN_MODEL_TESTS=1 uv run pytest tests/integration/test_hybrid_retrieval.py
"""

from __future__ import annotations

import os

import pytest

from app.ingestion.embedder import Embedder
from app.ingestion.index_builder import build_index
from app.retrieval.chunk_store import InMemoryChunkStore, StoredChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import HybridRetriever
from app.retrieval.sparse import SparseRetriever, build_bm25

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1 to download and run the model"
)

PASSAGES = [
    "The Transformer uses multi-head self-attention, allowing the model to jointly attend "
    "to information from different representation subspaces at different positions.",
    "Batch normalization reduces internal covariate shift by normalizing layer inputs over "
    "each training mini-batch, allowing much higher learning rates.",
    "Proximal Policy Optimization alternates between sampling data through interaction with "
    "the environment and optimizing a clipped surrogate objective.",
    "We build the vector store on IndexFlatIP, an exhaustive inner-product index; on "
    "unit-normalized embeddings its scores are exact cosine similarities.",
    "Low-rank adaptation freezes the pretrained weights and injects trainable rank "
    "decomposition matrices into each layer of the architecture.",
]


@pytest.fixture(scope="module")
def store() -> InMemoryChunkStore:
    return InMemoryChunkStore(
        StoredChunk(
            vector_id=i,
            text=text,
            page_number=i + 1,
            chunk_index=i,
            source_filename="corpus.pdf",
        )
        for i, text in enumerate(PASSAGES)
    )


@pytest.fixture(scope="module")
def dense(store: InMemoryChunkStore) -> DenseRetriever:
    embedder = Embedder()
    index = build_index(embedder.embed_documents(PASSAGES), list(range(len(PASSAGES))))
    return DenseRetriever(index, store, embedder)


@pytest.fixture(scope="module")
def sparse(store: InMemoryChunkStore) -> SparseRetriever:
    return SparseRetriever(build_bm25(PASSAGES, list(range(len(PASSAGES)))), store)


def test_dense_retrieval_handles_a_paraphrase_with_no_shared_terms(
    dense: DenseRetriever,
) -> None:
    """No content word here appears in passage 4 — only sparse would miss it."""
    hits = dense.retrieve("how are large models fine-tuned cheaply?", k=2)

    assert hits[0].vector_id == 4


def test_sparse_retrieval_handles_an_exact_identifier(sparse: SparseRetriever) -> None:
    hits = sparse.retrieve("IndexFlatIP", k=2)

    assert hits[0].vector_id == 3


def test_hybrid_answers_both_query_shapes(dense: DenseRetriever, sparse: SparseRetriever) -> None:
    hybrid = HybridRetriever([dense, sparse], candidate_k=5)

    assert hybrid.retrieve("how are large models fine-tuned cheaply?", k=2)[0].vector_id == 4
    assert hybrid.retrieve("IndexFlatIP", k=2)[0].vector_id == 3


def test_hybrid_hits_are_citable(dense: DenseRetriever, sparse: SparseRetriever) -> None:
    hit = HybridRetriever([dense, sparse], candidate_k=5).retrieve("clipped surrogate", k=1)[0]

    assert (hit.source_filename, hit.page_number, hit.vector_id) == ("corpus.pdf", 3, 2)
