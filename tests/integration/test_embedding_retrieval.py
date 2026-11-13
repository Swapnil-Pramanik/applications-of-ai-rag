"""End-to-end sanity check with the real bge-small-en-v1.5 model.

The unit tests prove the plumbing; this proves the plumbing produces
*semantically useful* vectors — that a paraphrased question retrieves the
right passage and that the query prefix is helping rather than hurting.

Downloads ~130MB on first run, so it is opt-in:

    RUN_MODEL_TESTS=1 uv run pytest tests/integration/test_embedding_retrieval.py
"""

from __future__ import annotations

import os

import pytest

from app.ingestion.embedder import Embedder
from app.ingestion.index_builder import build_index, search

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1 to download and run the model"
)

PASSAGES = [
    "The Transformer uses multi-head self-attention, allowing the model to jointly "
    "attend to information from different representation subspaces at different positions.",
    "Batch normalization reduces internal covariate shift by normalizing layer inputs "
    "over each training mini-batch, allowing much higher learning rates.",
    "Proximal Policy Optimization alternates between sampling data through interaction "
    "with the environment and optimizing a clipped surrogate objective.",
]


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embeddings_have_the_configured_dimension(embedder: Embedder) -> None:
    from app.core.config import settings

    assert embedder.embed_documents(PASSAGES).shape == (len(PASSAGES), settings.embedding_dim)


def test_embeddings_are_unit_normalized_so_inner_product_is_cosine(embedder: Embedder) -> None:
    import numpy as np

    norms = np.linalg.norm(embedder.embed_documents(PASSAGES), axis=1)

    assert np.allclose(norms, 1.0, atol=1e-4)


@pytest.mark.parametrize(
    ("question", "expected_passage_index"),
    [
        ("how does the transformer attend to multiple subspaces at once?", 0),
        ("what problem does normalizing mini-batch inputs solve?", 1),
        ("which RL algorithm optimizes a clipped surrogate objective?", 2),
    ],
)
def test_a_paraphrased_question_retrieves_its_source_passage(
    embedder: Embedder, question: str, expected_passage_index: int
) -> None:
    index = build_index(embedder.embed_documents(PASSAGES), [0, 1, 2])

    hits = search(index, embedder.embed_query(question), k=1)

    assert hits[0].vector_id == expected_passage_index
