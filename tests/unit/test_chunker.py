import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.parsers import PageContent


def test_chunk_pages_respects_size_budget() -> None:
    text = "word " * 500  # 2500 chars
    pages = [PageContent(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200


def test_chunk_pages_preserves_page_number() -> None:
    pages = [
        PageContent(page_number=1, text="Short page one text."),
        PageContent(page_number=2, text="Short page two text."),
    ]

    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=100)

    assert [c.page_number for c in chunks] == [1, 2]


def test_chunk_pages_assigns_sequential_chunk_index() -> None:
    pages = [PageContent(page_number=1, text="word " * 500)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_overlap_carries_context_between_chunks() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    pages = [PageContent(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=20, chunk_overlap=8)

    assert len(chunks) > 1
    # Each chunk after the first should open with a suffix of words carried
    # over from the end of the previous chunk.
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        prev_words = prev.text.split()
        curr_words = curr.text.split()
        assert any(prev_words[-k:] == curr_words[:k] for k in range(1, len(prev_words) + 1))


def test_chunk_pages_char_fallback_respects_size_on_unsplittable_text() -> None:
    # A single token with no separators at all forces the raw-character
    # fallback; every emitted chunk must still respect chunk_size.
    text = "x" * 500
    pages = [PageContent(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 50


def test_chunk_pages_rejects_overlap_ge_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages([PageContent(page_number=1, text="x")], chunk_size=100, chunk_overlap=100)


def test_chunk_pages_drops_empty_pages() -> None:
    pages = [PageContent(page_number=1, text="   \n\n  ")]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)

    assert chunks == []


def test_chunk_pages_splits_on_paragraph_boundary_first() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    pages = [PageContent(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=30, chunk_overlap=0)

    assert any("First paragraph" in c.text for c in chunks)
    assert any("Second paragraph" in c.text for c in chunks)
    assert any("Third paragraph" in c.text for c in chunks)
