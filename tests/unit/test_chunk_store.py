"""ChunkStore contract tests.

The one behaviour worth pinning is that a missing vector id is *dropped*, not
faked or raised on — a document deleted after its vectors were indexed must
degrade to a shorter result list, never to a wrong citation.
"""

from __future__ import annotations

from app.retrieval.chunk_store import ChunkStore, InMemoryChunkStore, StoredChunk


def make_chunk(vector_id: int, text: str = "text") -> StoredChunk:
    return StoredChunk(
        vector_id=vector_id,
        text=text,
        page_number=1,
        chunk_index=vector_id,
        source_filename="paper.pdf",
    )


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryChunkStore([]), ChunkStore)


def test_lookup_returns_the_requested_chunks_keyed_by_vector_id() -> None:
    store = InMemoryChunkStore([make_chunk(1, "alpha"), make_chunk(2, "beta")])

    assert store.get_many([2, 1]) == {1: make_chunk(1, "alpha"), 2: make_chunk(2, "beta")}


def test_unknown_vector_ids_are_dropped_rather_than_raising() -> None:
    store = InMemoryChunkStore([make_chunk(1)])

    assert store.get_many([1, 99]) == {1: make_chunk(1)}


def test_empty_store_has_no_chunks() -> None:
    store = InMemoryChunkStore([])

    assert len(store) == 0
    assert store.get_many([1]) == {}
