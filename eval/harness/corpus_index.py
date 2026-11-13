"""Builds the retrieval stack the harness measures, over the real 18-paper corpus.

**This deliberately reuses `app.ingestion.pipeline`.** The whole claim of this
project is that the committed numbers describe the deployed system; that is
only true if evaluation runs the same parse -> chunk -> embed -> index code
production runs. The one thing the harness swaps is the `ChunkStore` — there
is no tenant and no Postgres here, so `InMemoryChunkStore` stands in behind
the protocol both paths share.

**Why the on-disk cache.** Embedding is 91% of ingestion time (see
`bench/results/ingestion.json`), and the Day 9 sweep re-runs the whole corpus
once per chunking configuration. Caching by a hash of `(chunk_size,
chunk_overlap, embedding_model, corpus file mtimes+sizes)` turns a repeat
sweep from minutes into seconds while still rebuilding the moment any input
actually changes. The mtime/size fingerprint is what makes it safe to leave
on: swap a PDF and the cache key moves with it.

The cache holds only derived data and is gitignored — `make eval` from a clean
checkout rebuilds it, which is the reproducibility promise in CLAUDE.md §6.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.corpus import CORPUS
from app.ingestion.embedder import Vector, get_embedder
from app.ingestion.index_builder import build_index
from app.ingestion.pipeline import EmbeddedChunk, ingest_document
from app.retrieval.base import Retriever
from app.retrieval.chunk_store import InMemoryChunkStore, StoredChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import HybridRetriever
from app.retrieval.reranker import Reranker, RerankingRetriever
from app.retrieval.sparse import SparseRetriever, build_bm25

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "papers"
DEFAULT_CACHE_DIR = REPO_ROOT / "eval" / ".cache"

# Every arm of the ablation. A `+rerank` variant wraps the same base retriever
# in the cross-encoder, so each pair differs by exactly one pipeline stage.
STRATEGIES = (
    "dense",
    "sparse",
    "hybrid",
    "dense+rerank",
    "sparse+rerank",
    "hybrid+rerank",
)


class CorpusNotFoundError(FileNotFoundError):
    """Raised when the eval corpus has not been downloaded yet."""


@dataclass(frozen=True)
class EvalCorpus:
    """The ingested corpus plus everything a retriever needs to be built on it."""

    chunks: list[EmbeddedChunk]
    vectors: Vector
    chunk_size: int
    chunk_overlap: int

    @property
    def texts(self) -> list[str]:
        return [chunk.text for chunk in self.chunks]

    def chunk_store(self) -> InMemoryChunkStore:
        return InMemoryChunkStore(
            StoredChunk(
                vector_id=chunk.vector_id,
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                source_filename=chunk.source_filename,
            )
            for chunk in self.chunks
        )

    def build_retriever(
        self,
        strategy: str,
        *,
        candidate_k: int | None = None,
        rrf_k: int | None = None,
        reranker: Reranker | None = None,
    ) -> Retriever:
        """One of `STRATEGIES` — the arms of the Day 8 ablation.

        A `+rerank` suffix wraps the same base retriever in a cross-encoder
        rather than building a different one, so `hybrid` and `hybrid+rerank`
        differ by exactly one stage and the delta between them is attributable.

        Built per call rather than cached because the sweep varies `rrf_k` and
        `candidate_k`, and a retriever silently holding a stale parameter is
        the kind of bug that produces a plausible-looking wrong number.
        """
        base_name, _, suffix = strategy.partition("+")
        if suffix not in ("", "rerank"):
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

        store = self.chunk_store()
        vector_ids = [chunk.vector_id for chunk in self.chunks]

        base: Retriever
        if base_name == "dense":
            base = DenseRetriever(build_index(self.vectors, vector_ids), store)
        elif base_name == "sparse":
            base = SparseRetriever(build_bm25(self.texts, vector_ids), store)
        elif base_name == "hybrid":
            base = HybridRetriever(
                [
                    DenseRetriever(build_index(self.vectors, vector_ids), store),
                    SparseRetriever(build_bm25(self.texts, vector_ids), store),
                ],
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
        else:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

        if suffix == "rerank":
            return RerankingRetriever(base, reranker=reranker, candidate_k=candidate_k)
        return base


def corpus_paths(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Path]:
    """The 18 canonical PDFs, in a stable order so vector ids are reproducible.

    Sorted by arXiv id rather than directory order: `vector_id` is assigned by
    ingestion position, and golden-set debugging is much easier when the same
    chunk keeps the same id between runs.
    """
    paths = [corpus_dir / f"{arxiv_id}.pdf" for arxiv_id in sorted(CORPUS)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise CorpusNotFoundError(
            f"{len(missing)} of {len(paths)} corpus PDFs missing from {corpus_dir} "
            f"(first: {missing[0]}) — run `make corpus`"
        )
    return paths


def _fingerprint(paths: list[Path], chunk_size: int, chunk_overlap: int) -> str:
    """Hash every input that can change the ingested output."""
    parts = [settings.embedding_model, str(chunk_size), str(chunk_overlap)]
    for path in paths:
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _read_cache(cache_dir: Path, key: str) -> tuple[list[EmbeddedChunk], Vector] | None:
    meta_path, vectors_path = cache_dir / f"{key}.json", cache_dir / f"{key}.npy"
    if not (meta_path.exists() and vectors_path.exists()):
        return None

    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = [EmbeddedChunk(**row) for row in rows]
    vectors: Vector = np.load(vectors_path)
    # A truncated write would desync these; rebuilding is cheaper than
    # debugging a chunk/vector misalignment that only shows up as bad recall.
    if len(chunks) != len(vectors):
        return None
    return chunks, vectors


def _write_cache(cache_dir: Path, key: str, chunks: list[EmbeddedChunk], vectors: Vector) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps([chunk.__dict__ for chunk in chunks]), encoding="utf-8"
    )
    np.save(cache_dir / f"{key}.npy", vectors)


def load_corpus(
    *,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> EvalCorpus:
    """Ingest (or read from cache) the whole corpus at one chunking configuration.

    Pass `cache_dir=None` to force a rebuild — used by the test that proves the
    cache round-trips rather than merely that it exists.
    """
    paths = corpus_paths(corpus_dir)

    if cache_dir is not None:
        key = _fingerprint(paths, chunk_size, chunk_overlap)
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            chunks, vectors = cached
            return EvalCorpus(chunks, vectors, chunk_size, chunk_overlap)

    embedder = get_embedder()
    all_chunks: list[EmbeddedChunk] = []
    all_vectors: list[Vector] = []
    next_id = 0
    for path in paths:
        result = ingest_document(
            path,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_vector_id=next_id,
        )
        if not result.chunks:
            continue
        all_chunks.extend(result.chunks)
        all_vectors.append(result.vectors)
        next_id = result.next_vector_id

    vectors = (
        np.vstack(all_vectors)
        if all_vectors
        else np.empty((0, settings.embedding_dim), dtype="float32")
    )
    if cache_dir is not None:
        _write_cache(cache_dir, _fingerprint(paths, chunk_size, chunk_overlap), all_chunks, vectors)

    return EvalCorpus(all_chunks, vectors, chunk_size, chunk_overlap)
