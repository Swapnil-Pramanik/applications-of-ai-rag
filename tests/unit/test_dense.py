"""Dense retriever tests.

Real FAISS, synthetic unit vectors and a stub embedder — so these assert the
retriever's contract (ordering, hydration, query-side prefixing) without
loading a 130MB model or going near the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ingestion.embedder import QUERY_INSTRUCTION, Embedder
from app.ingestion.index_builder import build_index, new_index
from app.retrieval.chunk_store import InMemoryChunkStore, StoredChunk
from app.retrieval.dense import DenseRetriever

DIM = 4

# Three orthogonal unit vectors, so which one wins is unambiguous.
VECTORS = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
)


class StubEmbedder(Embedder):
    """Maps a query string straight onto one of the corpus vectors."""

    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.seen_queries: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:  # type: ignore[override]
        self.seen_queries.append(text)
        return self.vector


@pytest.fixture(autouse=True)
def toy_embedding_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.config.settings.embedding_dim", DIM)


@pytest.fixture
def store() -> InMemoryChunkStore:
    return InMemoryChunkStore(
        StoredChunk(
            vector_id=vector_id,
            text=f"chunk {vector_id}",
            page_number=vector_id + 1,
            chunk_index=vector_id,
            source_filename="paper.pdf",
        )
        for vector_id in (10, 11, 12)
    )


@pytest.fixture
def index():  # type: ignore[no-untyped-def]
    return build_index(VECTORS, [10, 11, 12], dim=DIM)


def test_the_nearest_vector_is_returned_first(index, store: InMemoryChunkStore) -> None:  # type: ignore[no-untyped-def]
    retriever = DenseRetriever(index, store, StubEmbedder(VECTORS[1]))

    assert retriever.retrieve("anything", k=3)[0].vector_id == 11


def test_hits_carry_the_citation_metadata_for_their_vector(
    index, store: InMemoryChunkStore
) -> None:  # type: ignore[no-untyped-def]
    hit = DenseRetriever(index, store, StubEmbedder(VECTORS[2])).retrieve("anything", k=1)[0]

    assert (hit.text, hit.page_number, hit.source_filename) == ("chunk 12", 13, "paper.pdf")


def test_scores_are_cosine_similarity_on_unit_vectors(index, store: InMemoryChunkStore) -> None:  # type: ignore[no-untyped-def]
    hit = DenseRetriever(index, store, StubEmbedder(VECTORS[0])).retrieve("anything", k=1)[0]

    assert hit.score == pytest.approx(1.0)


def test_faiss_ordering_is_preserved_not_re_sorted(index, store: InMemoryChunkStore) -> None:  # type: ignore[no-untyped-def]
    hits = DenseRetriever(index, store, StubEmbedder(VECTORS[1])).retrieve("anything", k=3)

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_k_larger_than_the_corpus_returns_what_exists(index, store: InMemoryChunkStore) -> None:  # type: ignore[no-untyped-def]
    assert len(DenseRetriever(index, store, StubEmbedder(VECTORS[0])).retrieve("q", k=50)) == 3


def test_an_empty_index_returns_nothing(store: InMemoryChunkStore) -> None:
    retriever = DenseRetriever(new_index(DIM), store, StubEmbedder(VECTORS[0]))

    assert retriever.retrieve("anything", k=5) == []


def test_ids_the_chunk_store_no_longer_knows_are_dropped(index) -> None:  # type: ignore[no-untyped-def]
    partial = InMemoryChunkStore(
        [StoredChunk(10, "chunk 10", page_number=1, chunk_index=0, source_filename="paper.pdf")]
    )

    hits = DenseRetriever(index, partial, StubEmbedder(VECTORS[0])).retrieve("q", k=3)

    assert [hit.vector_id for hit in hits] == [10]


def test_the_query_goes_through_the_bge_instruction_prefix(store: InMemoryChunkStore) -> None:
    """Guards the asymmetric-prefix contract from the embedder's docstring.

    Using `embed_documents` here instead would still return results — just
    quietly worse ones — so nothing but a test catches the regression.
    """
    stub_model_calls: list[str] = []

    class RecordingModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            stub_model_calls.extend(texts)
            return np.tile(VECTORS[0], (len(texts), 1))

    retriever = DenseRetriever(new_index(DIM), store, Embedder(model=RecordingModel()))
    retriever.retrieve("what is LoRA?", k=1)

    assert stub_model_calls == [QUERY_INSTRUCTION + "what is LoRA?"]
