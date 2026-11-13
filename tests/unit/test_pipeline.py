"""Ingestion pipeline tests, run with a stub embedder against fixture files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ingestion.embedder import Embedder
from app.ingestion.index_builder import search
from app.ingestion.pipeline import index_from_results, ingest_document

DIM = 4
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class StubModel:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        vectors = np.random.default_rng(0).standard_normal((len(texts), DIM)).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def toy_embedding_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import config

    monkeypatch.setattr(config.settings, "embedding_dim", DIM)


@pytest.fixture
def embedder() -> Embedder:
    return Embedder(model=StubModel())


def test_ingesting_a_document_produces_one_vector_per_chunk(embedder: Embedder) -> None:
    result = ingest_document(FIXTURES / "sample.md", embedder=embedder, chunk_size=200)

    assert len(result.chunks) == len(result.vectors)
    assert len(result.chunks) > 0


def test_chunks_carry_the_source_filename_for_citations(embedder: Embedder) -> None:
    result = ingest_document(FIXTURES / "sample.md", embedder=embedder, chunk_size=200)

    assert {chunk.source_filename for chunk in result.chunks} == {"sample.md"}


def test_vector_ids_are_contiguous_from_the_supplied_start(embedder: Embedder) -> None:
    result = ingest_document(FIXTURES / "sample.md", embedder=embedder, chunk_size=200)

    assert [chunk.vector_id for chunk in result.chunks] == list(range(len(result.chunks)))


def test_a_second_document_continues_numbering_instead_of_colliding(
    embedder: Embedder,
) -> None:
    first = ingest_document(FIXTURES / "sample.md", embedder=embedder, chunk_size=200)
    second = ingest_document(
        FIXTURES / "sample.txt",
        embedder=embedder,
        chunk_size=200,
        start_vector_id=first.next_vector_id,
    )

    first_ids = {chunk.vector_id for chunk in first.chunks}
    second_ids = {chunk.vector_id for chunk in second.chunks}
    assert first_ids.isdisjoint(second_ids)
    assert min(second_ids) == max(first_ids) + 1


def test_next_vector_id_of_an_empty_ingestion_is_zero(embedder: Embedder, tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    assert ingest_document(empty, embedder=embedder).next_vector_id == 0


def test_index_from_results_indexes_every_chunk_across_documents(embedder: Embedder) -> None:
    first = ingest_document(FIXTURES / "sample.md", embedder=embedder, chunk_size=200)
    second = ingest_document(
        FIXTURES / "sample.txt",
        embedder=embedder,
        chunk_size=200,
        start_vector_id=first.next_vector_id,
    )

    index = index_from_results([first, second])

    assert index.ntotal == len(first.chunks) + len(second.chunks)


def test_searching_the_built_index_returns_a_real_chunks_vector_id(embedder: Embedder) -> None:
    result = ingest_document(FIXTURES / "sample.md", embedder=embedder, chunk_size=200)

    hits = search(index_from_results([result]), result.vectors[0], k=1)

    assert hits[0].vector_id in {chunk.vector_id for chunk in result.chunks}


def test_a_corpus_of_empty_documents_yields_an_empty_index(
    embedder: Embedder, tmp_path: Path
) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    assert index_from_results([ingest_document(empty, embedder=embedder)]).ntotal == 0
