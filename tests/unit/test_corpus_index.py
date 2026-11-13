"""Tests for the harness's corpus fixture.

The cache is the only part here with teeth. It exists so the Day 9 sweep does
not re-embed 1,400 chunks per configuration, and its failure mode is nasty: a
stale hit returns *plausible* chunks built at the wrong `chunk_size`, so the
sweep silently reports the same numbers for every point on the grid. The
fingerprint tests are therefore the real subject of this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.ingestion.pipeline import EmbeddedChunk
from app.retrieval.reranker import Reranker, RerankingRetriever
from eval.harness.corpus_index import (
    STRATEGIES,
    CorpusNotFoundError,
    EvalCorpus,
    _fingerprint,
    _read_cache,
    _write_cache,
    corpus_paths,
)


def make_corpus(n: int = 3) -> EvalCorpus:
    chunks = [
        EmbeddedChunk(
            vector_id=i,
            chunk_index=i,
            page_number=i + 1,
            text=f"chunk {i}",
            source_filename="1706.03762.pdf",
        )
        for i in range(n)
    ]
    # Real embedding width, not a toy one: `build_index` asserts the vectors
    # match `settings.embedding_dim`, so a 4-dim fixture can't build the dense
    # arms at all.
    return EvalCorpus(chunks, np.ones((n, settings.embedding_dim), dtype="float32"), 1000, 150)


# --- corpus discovery ------------------------------------------------------


def test_a_missing_corpus_names_the_fix_rather_than_failing_obscurely(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotFoundError, match="make corpus"):
        corpus_paths(tmp_path)


# --- cache fingerprinting --------------------------------------------------


def test_the_fingerprint_changes_when_chunk_size_changes(tmp_path: Path) -> None:
    """The bug this prevents: a whole sweep reporting one configuration's numbers."""
    pdf = tmp_path / "1706.03762.pdf"
    pdf.write_bytes(b"pdf")

    assert _fingerprint([pdf], 1000, 150) != _fingerprint([pdf], 500, 150)


def test_the_fingerprint_changes_when_chunk_overlap_changes(tmp_path: Path) -> None:
    pdf = tmp_path / "1706.03762.pdf"
    pdf.write_bytes(b"pdf")

    assert _fingerprint([pdf], 1000, 150) != _fingerprint([pdf], 1000, 0)


def test_the_fingerprint_changes_when_a_source_pdf_changes(tmp_path: Path) -> None:
    """Swapping a PDF must invalidate the cache — this is what makes it safe to leave on."""
    pdf = tmp_path / "1706.03762.pdf"
    pdf.write_bytes(b"pdf")
    before = _fingerprint([pdf], 1000, 150)

    pdf.write_bytes(b"a completely different pdf")

    assert _fingerprint([pdf], 1000, 150) != before


def test_the_fingerprint_is_stable_for_unchanged_inputs(tmp_path: Path) -> None:
    pdf = tmp_path / "1706.03762.pdf"
    pdf.write_bytes(b"pdf")

    assert _fingerprint([pdf], 1000, 150) == _fingerprint([pdf], 1000, 150)


# --- cache round trip ------------------------------------------------------


def test_cache_round_trips_chunks_and_vectors(tmp_path: Path) -> None:
    corpus = make_corpus()
    _write_cache(tmp_path, "key", corpus.chunks, corpus.vectors)

    cached = _read_cache(tmp_path, "key")

    assert cached is not None
    chunks, vectors = cached
    assert chunks == corpus.chunks
    assert np.array_equal(vectors, corpus.vectors)


def test_a_missing_cache_entry_reads_as_none(tmp_path: Path) -> None:
    assert _read_cache(tmp_path, "never-written") is None


def test_a_desynced_cache_entry_is_rejected_rather_than_returned(tmp_path: Path) -> None:
    """A truncated write must force a rebuild, not a chunk/vector misalignment."""
    corpus = make_corpus(3)
    _write_cache(
        tmp_path, "key", corpus.chunks, np.ones((2, settings.embedding_dim), dtype="float32")
    )

    assert _read_cache(tmp_path, "key") is None


# --- retriever construction ------------------------------------------------


def test_an_unknown_strategy_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        make_corpus().build_retriever("magic")


def test_an_unknown_suffix_is_rejected_rather_than_silently_ignored() -> None:
    """`hybrid+rerankk` must fail loudly, not quietly evaluate plain hybrid and
    get recorded as a reranked result."""
    with pytest.raises(ValueError, match="unknown strategy"):
        make_corpus().build_retriever("hybrid+rerankk")


def test_a_rerank_suffix_wraps_the_base_rather_than_replacing_it() -> None:
    """The ablation is only attributable if `x` and `x+rerank` share a base."""
    corpus = make_corpus()

    plain = corpus.build_retriever("sparse")
    reranked = corpus.build_retriever("sparse+rerank", reranker=Reranker(model=object()))

    assert isinstance(reranked, RerankingRetriever)
    assert not isinstance(plain, RerankingRetriever)


def test_every_advertised_strategy_can_actually_be_built() -> None:
    """STRATEGIES is what the CLI offers; an unbuildable entry is a broken flag."""
    corpus = make_corpus()

    for strategy in STRATEGIES:
        assert corpus.build_retriever(strategy, reranker=Reranker(model=object())) is not None


def test_the_chunk_store_exposes_every_ingested_chunk() -> None:
    corpus = make_corpus(3)

    store = corpus.chunk_store()

    assert len(store) == 3
    assert store.get_many([1])[1].page_number == 2
