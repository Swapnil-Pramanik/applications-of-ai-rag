"""Per-tenant FAISS vector indexes.

**One index file per tenant** (`<index_dir>/<user_id>.faiss`), mirroring the
row-level-security boundary already enforced in Postgres. A tenant's vectors
are never in another tenant's index, so cross-tenant leakage is structurally
impossible rather than dependent on an application filter being remembered at
every call site. It also keeps each search over a small index and lets a cold
Cloud Run instance load only the tenant it was woken for.

Index type: `IndexIDMap2(IndexFlatIP)`.

- `IndexFlatIP` is exhaustive inner-product search. On unit-normalized vectors
  (see `embedder`) inner product *is* cosine similarity, so recall is exact —
  no approximation error to confound the Day 6-9 eval numbers. At the
  designed ceiling of ~50k chunks x 384 dims that is ~77MB and a brute-force
  scan in single-digit milliseconds, which is why the plan defers
  `IndexIVFFlat` until that ceiling is actually crossed.
- `IndexIDMap2` lets us attach our own int64 ids instead of relying on
  insertion position. Those ids are what `chunks.vector_id` stores, so
  positions surviving a rebuild is not something we have to guarantee, and a
  future `remove_ids` can delete a document's vectors without renumbering
  everything after it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from app.core.config import settings
from app.ingestion.embedder import Vector


class IndexDimensionMismatchError(ValueError):
    """A persisted index's dimension disagrees with the configured embedding dim."""


@dataclass(frozen=True)
class SearchHit:
    vector_id: int
    score: float


def index_path(user_id: UUID | str, index_dir: Path | None = None) -> Path:
    return (index_dir or settings.index_dir) / f"{user_id}.faiss"


def new_index(dim: int | None = None) -> faiss.Index:
    dim = dim if dim is not None else settings.embedding_dim
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))


def build_index(vectors: Vector, vector_ids: list[int], dim: int | None = None) -> faiss.Index:
    """Build a fresh index from vectors and their caller-assigned int64 ids."""
    dim = dim if dim is not None else settings.embedding_dim
    if len(vectors) != len(vector_ids):
        raise ValueError(f"got {len(vectors)} vectors but {len(vector_ids)} ids")

    index = new_index(dim)
    if len(vector_ids) == 0:
        return index
    if vectors.shape[1] != dim:
        raise IndexDimensionMismatchError(f"vectors are {vectors.shape[1]}-dim, index is {dim}-dim")

    index.add_with_ids(
        np.ascontiguousarray(vectors, dtype=np.float32),
        np.asarray(vector_ids, dtype=np.int64),
    )
    return index


def save_index(index: faiss.Index, user_id: UUID | str, index_dir: Path | None = None) -> Path:
    path = index_path(user_id, index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a crash mid-write leaves the previous index intact
    # rather than a truncated file that fails to load on next cold start.
    tmp_path = path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(path)
    return path


def load_index(user_id: UUID | str, index_dir: Path | None = None) -> faiss.Index:
    """Load a tenant's index, or an empty one if they have never ingested."""
    path = index_path(user_id, index_dir)
    if not path.exists():
        return new_index()

    index = faiss.read_index(str(path))
    if index.d != settings.embedding_dim:
        raise IndexDimensionMismatchError(
            f"{path} is {index.d}-dim but embedding_dim is {settings.embedding_dim}; "
            "the embedding model changed — re-ingest to rebuild the index"
        )
    return index


def search(index: faiss.Index, query_vector: Vector, k: int) -> list[SearchHit]:
    if index.ntotal == 0 or k <= 0:
        return []

    scores, ids = index.search(np.ascontiguousarray([query_vector], dtype=np.float32), k)
    # FAISS pads with id -1 when the index holds fewer than k vectors.
    return [
        SearchHit(vector_id=int(vid), score=float(score))
        for vid, score in zip(ids[0], scores[0], strict=True)
        if vid != -1
    ]


class TenantIndexCache:
    """LRU cache of loaded tenant indexes.

    Cloud Run instances are memory-capped, so we bound how many tenants are
    resident at once instead of letting a busy instance accumulate every index
    it has ever served.
    """

    def __init__(self, max_size: int | None = None, index_dir: Path | None = None):
        self._max_size = max_size if max_size is not None else settings.index_cache_size
        self._index_dir = index_dir
        self._cache: OrderedDict[str, faiss.Index] = OrderedDict()

    def get(self, user_id: UUID | str) -> faiss.Index:
        key = str(user_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        index = load_index(key, self._index_dir)
        self._cache[key] = index
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return index

    def invalidate(self, user_id: UUID | str) -> None:
        """Drop a tenant's cached index — call after re-ingesting for them."""
        self._cache.pop(str(user_id), None)
