"""Retrieval strategies, all behind the single `Retriever` interface.

`dense` and `sparse` are the two independent notions of relevance; `fusion`
combines them with reciprocal rank fusion. Day 8 swaps between all three to
measure whether hybrid actually earns its cost.
"""

from app.retrieval.base import RetrievedChunk, Retriever
from app.retrieval.chunk_store import ChunkStore, InMemoryChunkStore, StoredChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.sparse import BM25Index, SparseRetriever, build_bm25, load_bm25, save_bm25

__all__ = [
    "BM25Index",
    "ChunkStore",
    "DenseRetriever",
    "HybridRetriever",
    "InMemoryChunkStore",
    "RetrievedChunk",
    "Retriever",
    "SparseRetriever",
    "StoredChunk",
    "build_bm25",
    "load_bm25",
    "reciprocal_rank_fusion",
    "save_bm25",
]
