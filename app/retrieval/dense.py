"""Dense semantic retrieval over a tenant's FAISS index."""

from __future__ import annotations

import faiss

from app.ingestion.embedder import Embedder, get_embedder
from app.ingestion.index_builder import search
from app.retrieval.base import RetrievedChunk
from app.retrieval.chunk_store import ChunkStore


class DenseRetriever:
    """Embeds the query, searches FAISS, hydrates hits from the chunk store.

    Note the query goes through `embed_query`, not `embed_documents` — BGE
    puts its retrieval instruction on the query side only, and skipping it
    costs recall without raising anything (see `app.ingestion.embedder`).
    """

    # Cosine similarity: comparable within one query's results, not between
    # queries — it drifts with query length and phrasing.
    emits_calibrated_scores = False

    def __init__(
        self,
        index: faiss.Index,
        chunk_store: ChunkStore,
        embedder: Embedder | None = None,
    ):
        self._index = index
        self._chunk_store = chunk_store
        self._embedder = embedder or get_embedder()

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = search(self._index, self._embedder.embed_query(query), k)
        if not hits:
            return []

        chunks = self._chunk_store.get_many(hit.vector_id for hit in hits)
        # FAISS already returned these in descending-score order; preserve it
        # rather than re-sorting, and drop ids the store no longer knows.
        return [
            RetrievedChunk.from_stored(chunks[hit.vector_id], hit.score)
            for hit in hits
            if hit.vector_id in chunks
        ]
