"""FAISS index tests.

These use real FAISS (it's cheap and offline) but synthetic unit vectors, so
they test the index contract — id round-tripping, cosine semantics, tenant
isolation, persistence — without depending on the embedding model.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from app.ingestion.index_builder import (
    IndexDimensionMismatchError,
    TenantIndexCache,
    build_index,
    index_path,
    load_index,
    save_index,
    search,
)

DIM = 4


@pytest.fixture(autouse=True)
def toy_embedding_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests index 4-dim toy vectors, so the configured dim must agree."""
    from app.core import config

    monkeypatch.setattr(config.settings, "embedding_dim", DIM)


def unit_vectors(rows: list[list[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


@pytest.fixture
def vectors() -> np.ndarray:
    return unit_vectors([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


def test_build_index_holds_every_vector(vectors: np.ndarray) -> None:
    index = build_index(vectors, [10, 20, 30], dim=DIM)

    assert index.ntotal == 3


def test_search_returns_caller_assigned_ids_not_positions(vectors: np.ndarray) -> None:
    """The whole point of IndexIDMap2: hits come back as chunks.vector_id."""
    index = build_index(vectors, [100, 200, 300], dim=DIM)

    hits = search(index, vectors[1], k=1)

    assert [hit.vector_id for hit in hits] == [200]


def test_scores_are_cosine_similarity_on_normalized_vectors(vectors: np.ndarray) -> None:
    index = build_index(vectors, [1, 2, 3], dim=DIM)

    hits = search(index, vectors[0], k=3)

    assert hits[0].score == pytest.approx(1.0)  # identical vector
    assert all(hit.score == pytest.approx(0.0) for hit in hits[1:])  # orthogonal


def test_search_ranks_by_descending_similarity() -> None:
    vectors = unit_vectors([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]])
    index = build_index(vectors, [1, 2, 3], dim=DIM)

    hits = search(index, unit_vectors([[1, 0, 0, 0]])[0], k=3)

    assert [hit.vector_id for hit in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_drops_faiss_padding_when_k_exceeds_index_size(vectors: np.ndarray) -> None:
    """FAISS pads short result sets with id -1; those must never reach a caller."""
    index = build_index(vectors, [1, 2, 3], dim=DIM)

    hits = search(index, vectors[0], k=10)

    assert len(hits) == 3
    assert all(hit.vector_id != -1 for hit in hits)


def test_searching_an_empty_index_returns_nothing() -> None:
    index = build_index(np.empty((0, DIM), dtype=np.float32), [], dim=DIM)

    assert search(index, np.zeros(DIM, dtype=np.float32), k=5) == []


def test_build_index_rejects_mismatched_vector_and_id_counts(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError, match="3 vectors but 2 ids"):
        build_index(vectors, [1, 2], dim=DIM)


def test_build_index_rejects_vectors_of_the_wrong_dimension(vectors: np.ndarray) -> None:
    with pytest.raises(IndexDimensionMismatchError):
        build_index(vectors, [1, 2, 3], dim=DIM + 1)


def test_index_survives_a_save_load_round_trip(tmp_path: Path, vectors: np.ndarray) -> None:
    user_id = uuid.uuid4()
    save_index(build_index(vectors, [7, 8, 9], dim=DIM), user_id, index_dir=tmp_path)

    reloaded = load_index(user_id, index_dir=tmp_path)

    assert reloaded.ntotal == 3
    assert search(reloaded, vectors[2], k=1)[0].vector_id == 9


def test_save_leaves_no_temp_file_behind(tmp_path: Path, vectors: np.ndarray) -> None:
    save_index(build_index(vectors, [1, 2, 3], dim=DIM), uuid.uuid4(), index_dir=tmp_path)

    assert list(tmp_path.glob("*.tmp")) == []


def test_a_tenant_with_no_index_yet_gets_an_empty_one_not_an_error(tmp_path: Path) -> None:
    index = load_index(uuid.uuid4(), index_dir=tmp_path)

    assert index.ntotal == 0


def test_tenants_get_separate_index_files(tmp_path: Path, vectors: np.ndarray) -> None:
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    save_index(build_index(vectors, [1, 2, 3], dim=DIM), user_a, index_dir=tmp_path)
    save_index(build_index(vectors[:1], [99], dim=DIM), user_b, index_dir=tmp_path)

    assert index_path(user_a, tmp_path) != index_path(user_b, tmp_path)
    assert load_index(user_a, index_dir=tmp_path).ntotal == 3
    assert load_index(user_b, index_dir=tmp_path).ntotal == 1


def test_one_tenants_vectors_are_unreachable_from_anothers_index(
    tmp_path: Path, vectors: np.ndarray
) -> None:
    """The isolation guarantee: no filter is applied, yet nothing leaks."""
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    save_index(build_index(vectors, [1, 2, 3], dim=DIM), user_a, index_dir=tmp_path)
    save_index(build_index(vectors[:1], [99], dim=DIM), user_b, index_dir=tmp_path)

    hits = search(load_index(user_b, index_dir=tmp_path), vectors[1], k=10)

    assert {hit.vector_id for hit in hits} == {99}


def test_loading_an_index_of_the_wrong_dimension_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the embedding model must not silently search a stale index."""
    from app.core import config

    user_id = uuid.uuid4()
    save_index(build_index(unit_vectors([[1, 0, 0, 0]]), [1], dim=DIM), user_id, index_dir=tmp_path)
    monkeypatch.setattr(config.settings, "embedding_dim", DIM + 1)

    with pytest.raises(IndexDimensionMismatchError, match="embedding model changed"):
        load_index(user_id, index_dir=tmp_path)


class TestTenantIndexCache:
    def test_repeated_gets_return_the_same_loaded_object(
        self, tmp_path: Path, vectors: np.ndarray
    ) -> None:
        user_id = uuid.uuid4()
        save_index(build_index(vectors, [1, 2, 3], dim=DIM), user_id, index_dir=tmp_path)
        cache = TenantIndexCache(max_size=2, index_dir=tmp_path)

        assert cache.get(user_id) is cache.get(user_id)

    def test_cache_evicts_the_least_recently_used_tenant(self, tmp_path: Path) -> None:
        users = [uuid.uuid4() for _ in range(3)]
        cache = TenantIndexCache(max_size=2, index_dir=tmp_path)

        first = cache.get(users[0])
        cache.get(users[1])
        cache.get(users[2])  # evicts users[0], the least recently used

        assert cache.get(users[0]) is not first

    def test_recent_use_protects_a_tenant_from_eviction(self, tmp_path: Path) -> None:
        users = [uuid.uuid4() for _ in range(3)]
        cache = TenantIndexCache(max_size=2, index_dir=tmp_path)

        first = cache.get(users[0])
        cache.get(users[1])
        cache.get(users[0])  # refresh users[0]
        cache.get(users[2])  # evicts users[1] instead

        assert cache.get(users[0]) is first

    def test_invalidate_forces_a_reload_after_re_ingestion(
        self, tmp_path: Path, vectors: np.ndarray
    ) -> None:
        user_id = uuid.uuid4()
        save_index(build_index(vectors[:1], [1], dim=DIM), user_id, index_dir=tmp_path)
        cache = TenantIndexCache(max_size=2, index_dir=tmp_path)
        assert cache.get(user_id).ntotal == 1

        save_index(build_index(vectors, [1, 2, 3], dim=DIM), user_id, index_dir=tmp_path)
        cache.invalidate(user_id)

        assert cache.get(user_id).ntotal == 3

    def test_invalidating_an_uncached_tenant_is_a_no_op(self, tmp_path: Path) -> None:
        TenantIndexCache(max_size=2, index_dir=tmp_path).invalidate(uuid.uuid4())
