"""BM25 sparse retrieval tests.

Two things here are genuinely load-bearing and would fail silently otherwise:
the tokenizer keeping notation like `bge-small-en-v1.5` whole (that's the
entire reason sparse retrieval is in this system), and index-time and
query-time tokenization staying identical.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.retrieval.chunk_store import InMemoryChunkStore, StoredChunk
from app.retrieval.sparse import (
    SparseRetriever,
    bm25_path,
    build_bm25,
    load_bm25,
    save_bm25,
    tokenize,
)

CORPUS = {
    0: "The Transformer uses multi-head self-attention instead of recurrence.",
    1: "We evaluate bge-small-en-v1.5 embeddings against GPT-3 baselines.",
    2: "IndexFlatIP performs exhaustive inner-product search over unit vectors.",
    3: "Batch normalization reduces internal covariate shift during training.",
}


@pytest.fixture
def store() -> InMemoryChunkStore:
    return InMemoryChunkStore(
        StoredChunk(
            vector_id=vector_id,
            text=text,
            page_number=vector_id + 1,
            chunk_index=vector_id,
            source_filename="paper.pdf",
        )
        for vector_id, text in CORPUS.items()
    )


@pytest.fixture
def retriever(store: InMemoryChunkStore) -> SparseRetriever:
    index = build_bm25(list(CORPUS.values()), list(CORPUS))
    return SparseRetriever(index, store)


# --- tokenizer -------------------------------------------------------------


def test_hyphenated_and_dotted_identifiers_survive_as_single_tokens() -> None:
    assert tokenize("bge-small-en-v1.5 and GPT-3 at 2.0") == [
        "bge-small-en-v1.5",
        "gpt-3",
        "2.0",
    ]


def test_stopwords_are_dropped_so_they_cannot_earn_a_rank_one_hit() -> None:
    assert tokenize("how are they doing it") == []


def test_paper_titles_are_not_gutted_by_the_stopword_list() -> None:
    """`all`, `is`, `you` and `need` are deliberately not stopwords."""
    assert tokenize("Attention Is All You Need") == ["attention", "all", "need"]


def test_punctuation_is_stripped_and_case_is_folded() -> None:
    assert tokenize("IndexFlatIP(dim), exact!") == ["indexflatip", "dim", "exact"]


def test_trailing_separators_do_not_leak_into_tokens() -> None:
    assert tokenize("end. next- item") == ["end", "next", "item"]


# --- index building --------------------------------------------------------


def test_an_empty_corpus_has_no_index_rather_than_a_broken_one() -> None:
    assert build_bm25([], []) is None


def test_mismatched_texts_and_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="2 texts but 1 ids"):
        build_bm25(["a", "b"], [0])


# --- retrieval -------------------------------------------------------------


def test_exact_notation_match_outranks_topical_similarity(retriever: SparseRetriever) -> None:
    hits = retriever.retrieve("bge-small-en-v1.5", k=3)

    assert hits[0].vector_id == 1


def test_hits_are_hydrated_with_citation_metadata(retriever: SparseRetriever) -> None:
    hit = retriever.retrieve("covariate shift", k=1)[0]

    assert hit.source_filename == "paper.pdf"
    assert hit.page_number == 4
    assert hit.text == CORPUS[3]


def test_hits_come_back_in_descending_score_order(retriever: SparseRetriever) -> None:
    hits = retriever.retrieve("transformer search unit vectors", k=4)

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_documents_sharing_no_query_term_are_not_returned(retriever: SparseRetriever) -> None:
    hits = retriever.retrieve("covariate", k=4)

    assert [hit.vector_id for hit in hits] == [3]


def test_keeping_identifiers_whole_means_a_bare_substring_does_not_match(
    retriever: SparseRetriever,
) -> None:
    """The cost of the identifier-preserving tokenizer, asserted deliberately.

    `self-attention` is one token, so a query for bare `attention` misses it.
    That is the trade we accept: exact notation matching is what sparse
    retrieval is here for, and the soft-synonym case is precisely what the
    dense retriever covers. It is also the clearest single argument for why
    this system fuses the two instead of picking one.
    """
    assert retriever.retrieve("attention", k=5) == []


def test_a_stopword_only_overlap_produces_no_hit_at_all(retriever: SparseRetriever) -> None:
    """Regression: this used to hand fusion a rank-1 hit earned by "are".

    The query below shares exactly one token with the corpus under a
    stopword-free tokenizer, and it is a function word. BM25 happily scored
    it, RRF saw a rank-1 sparse hit and promoted the wrong passage above the
    one dense had ranked correctly. Nothing here may match.
    """
    assert retriever.retrieve("how are large models fine-tuned cheaply?", k=5) == []


def test_a_query_with_no_usable_tokens_returns_nothing(retriever: SparseRetriever) -> None:
    assert retriever.retrieve("!!! ???", k=5) == []


def test_a_tenant_with_no_sparse_index_returns_nothing(store: InMemoryChunkStore) -> None:
    assert SparseRetriever(None, store).retrieve("transformer", k=5) == []


def test_k_is_respected(retriever: SparseRetriever) -> None:
    assert len(retriever.retrieve("attention search training embeddings", k=2)) == 2


def test_non_positive_k_returns_nothing(retriever: SparseRetriever) -> None:
    assert retriever.retrieve("transformer", k=0) == []


def test_ids_the_chunk_store_no_longer_knows_are_dropped() -> None:
    index = build_bm25(list(CORPUS.values()), list(CORPUS))
    partial = InMemoryChunkStore(
        [StoredChunk(0, CORPUS[0], page_number=1, chunk_index=0, source_filename="paper.pdf")]
    )

    hits = SparseRetriever(index, partial).retrieve("transformer covariate", k=5)

    assert [hit.vector_id for hit in hits] == [0]


# --- persistence -----------------------------------------------------------


def test_index_round_trips_through_disk(tmp_path: Path, store: InMemoryChunkStore) -> None:
    user_id = uuid.uuid4()
    index = build_bm25(list(CORPUS.values()), list(CORPUS))
    assert index is not None

    save_bm25(index, user_id, tmp_path)
    loaded = load_bm25(user_id, tmp_path)

    assert loaded is not None
    assert loaded.vector_ids == index.vector_ids
    assert SparseRetriever(loaded, store).retrieve("multi-head", k=1)[0].vector_id == 0


def test_a_tenant_who_never_ingested_loads_as_none(tmp_path: Path) -> None:
    assert load_bm25(uuid.uuid4(), tmp_path) is None


def test_each_tenant_gets_their_own_file(tmp_path: Path) -> None:
    one, two = uuid.uuid4(), uuid.uuid4()

    assert bm25_path(one, tmp_path) != bm25_path(two, tmp_path)


def test_saving_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    user_id = uuid.uuid4()
    index = build_bm25(list(CORPUS.values()), list(CORPUS))
    assert index is not None

    save_bm25(index, user_id, tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [f"{user_id}.bm25"]
