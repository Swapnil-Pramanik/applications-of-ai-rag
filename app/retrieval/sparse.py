"""Sparse keyword retrieval with BM25.

This exists because dense embeddings are lossy in exactly the places a
research-paper corpus is dense with meaning: `IndexIVFFlat`, `bge-small`,
`O(n log n)`, `GPT-3`, a section number, an author's surname. A 384-dim
sentence embedding compresses those into a general topical direction; BM25
matches the literal token. Neither is sufficient alone, which is the argument
`fusion.py` is built to test.

**Tokenization is the load-bearing detail.** A naive `text.split()` shreds the
very tokens hybrid retrieval is supposed to rescue: `bge-small-en-v1.5` would
become one unmatchable blob, `Flat, IP)` would keep its punctuation. The
tokenizer here lowercases and keeps runs of alphanumerics joined by internal
hyphens or dots, so `bge-small-en-v1.5`, `gpt-3` and `2.0` each survive as one
token. It lives in the same module as the index builder and the retriever on
purpose: query-time and index-time tokenization must be identical, and the
cheapest way to guarantee that is to make them impossible to edit separately.

**Stopwords are removed**, and that is a fusion-safety measure rather than an
efficiency one. BM25 scores every term it matches; a query sharing nothing
with a passage but the word "are" still produces a hit, and if it is the only
hit it is a *rank-1* hit. RRF reads ranks, not scores (see `fusion`), so it
cannot tell that rank-1 apart from one earned by matching `IndexFlatIP` and
promotes the passage as if two retrievers agreed. On a large corpus BM25's
negative-IDF floor mostly defuses this, since a stopword ends up in more than
half the chunks and is weighted to nothing — but relying on that means the
behaviour depends on corpus size, and the numbers this project reports are
supposed to describe the system, not the corpus that happened to be indexed.
The trade: an all-stopword query ("what is it") returns no sparse results at
all. That is the right answer, and the dense retriever still covers it.

**Persistence.** `rank_bm25` is a pure in-memory object with no save format,
so a tenant's index is pickled to `<index_dir>/<user_id>.bm25`, sitting beside
their `.faiss`. Same one-file-per-tenant reasoning as `index_builder`: tenant
isolation is structural, and a cold Cloud Run instance loads only what it was
woken for. The cost of that choice is a **sync invariant** — a tenant's
`.faiss` and `.bm25` must be rebuilt from the same chunk list in the same
pass, or the two retrievers will disagree about what vector id 41 means. Both
builders therefore take the identical `(texts | vectors, vector_ids)` shape,
and the pickle records the ids it was built from rather than relying on
corpus position, so a stale artifact yields missing lookups rather than
confidently wrong citations.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.retrieval.base import RetrievedChunk
from app.retrieval.chunk_store import ChunkStore

# Runs of alphanumerics, optionally joined by single internal hyphens or dots.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

# A deliberately small, closed English stopword list — function words only.
# Kept inline rather than pulled from NLTK to avoid a runtime corpus download
# on a Cloud Run cold start, and kept short on purpose: anything that could
# plausibly carry meaning in an ML paper stays in. Note "all", "is", "you" and
# "need" are absent, so "Attention Is All You Need" survives as a query.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    as into over under about between during
    be am is are was were been being do does did doing have has had having
    will would can could should may might must
    it its they them their we our us he she his her you your i me my
    there here what which who whom whose when where why how
    not no nor so such own same too very just also
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split keeping hyphenated/dotted identifiers, drop stopwords.

    Index-time and query-time both call this. They must never diverge — a
    token filtered on one side only silently stops matching.
    """
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


@dataclass(frozen=True)
class BM25Index:
    """A tenant's BM25 model plus the vector ids its corpus positions map to.

    `vector_ids[i]` is the id of corpus document `i`. Keeping the mapping
    inside the artifact means BM25 corpus positions never leak out of this
    module — every retriever speaks in vector ids.
    """

    bm25: BM25Okapi
    vector_ids: list[int]


def bm25_path(user_id: UUID | str, index_dir: Path | None = None) -> Path:
    return (index_dir or settings.index_dir) / f"{user_id}.bm25"


def build_bm25(texts: list[str], vector_ids: list[int]) -> BM25Index | None:
    """Build a tenant's BM25 index. Returns None for an empty corpus.

    `rank_bm25` divides by the corpus average document length, so it cannot be
    constructed over zero documents — an empty tenant simply has no sparse
    index, and `SparseRetriever` treats that as "no results".
    """
    if len(texts) != len(vector_ids):
        raise ValueError(f"got {len(texts)} texts but {len(vector_ids)} ids")
    if not texts:
        return None

    return BM25Index(bm25=BM25Okapi([tokenize(text) for text in texts]), vector_ids=vector_ids)


def save_bm25(index: BM25Index, user_id: UUID | str, index_dir: Path | None = None) -> Path:
    path = bm25_path(user_id, index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, matching `index_builder.save_index`: a crash mid-write
    # must not leave a truncated pickle that fails to load on cold start.
    tmp_path = path.with_suffix(".bm25.tmp")
    tmp_path.write_bytes(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
    tmp_path.replace(path)
    return path


def load_bm25(user_id: UUID | str, index_dir: Path | None = None) -> BM25Index | None:
    """Load a tenant's BM25 index, or None if they have never ingested.

    Unpickling is safe here only because these files are written by
    `save_bm25` into a directory the service owns; never point `index_dir` at
    user-supplied storage.
    """
    path = bm25_path(user_id, index_dir)
    if not path.exists():
        return None

    index: BM25Index = pickle.loads(path.read_bytes())
    return index


class SparseRetriever:
    # BM25 sums unbounded term weights, so magnitude depends on corpus
    # statistics and query length rather than on relevance.
    emits_calibrated_scores = False

    def __init__(self, index: BM25Index | None, chunk_store: ChunkStore):
        self._index = index
        self._chunk_store = chunk_store

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if self._index is None or not tokens or k <= 0:
            return []

        scores = self._index.bm25.get_scores(tokens)
        # A zero score means the document shares no query term at all. BM25
        # returns those anyway to fill out k; keeping them would pad the
        # fusion candidate list with documents that matched nothing, which
        # inflates their reciprocal rank for free.
        ranked = sorted(
            (
                (vector_id, float(score))
                for vector_id, score in zip(self._index.vector_ids, scores, strict=True)
                if score > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:k]

        chunks = self._chunk_store.get_many(vector_id for vector_id, _ in ranked)
        return [
            RetrievedChunk.from_stored(chunks[vector_id], score)
            for vector_id, score in ranked
            if vector_id in chunks
        ]
