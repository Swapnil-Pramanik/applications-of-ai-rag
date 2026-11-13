from pathlib import Path

import docx
import pymupdf
import pytest

from app.ingestion.parsers import (
    UnsupportedFormatError,
    parse_document,
    parse_docx,
    parse_pdf,
    parse_text,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_parse_text_txt() -> None:
    pages = parse_text(FIXTURES / "sample.txt")
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "plain text fixture" in pages[0].text


def test_parse_text_md() -> None:
    pages = parse_text(FIXTURES / "sample.md")
    assert len(pages) == 1
    assert "# Sample Heading" in pages[0].text


def test_parse_docx(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    document.save(str(doc_path))

    pages = parse_docx(doc_path)
    assert len(pages) == 1
    assert "First paragraph." in pages[0].text
    assert "Second paragraph." in pages[0].text


def test_parse_pdf_multi_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for label in ("Page one text.", "Page two text."):
        page = doc.new_page()
        page.insert_text((72, 72), label)
    doc.save(str(pdf_path))
    doc.close()

    pages = parse_pdf(pdf_path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Page one text." in pages[0].text
    assert pages[1].page_number == 2
    assert "Page two text." in pages[1].text


def test_parse_document_dispatches_by_extension() -> None:
    pages = parse_document(FIXTURES / "sample.txt")
    assert "plain text fixture" in pages[0].text


def test_parse_document_unsupported_format(tmp_path: Path) -> None:
    bad_path = tmp_path / "sample.xyz"
    bad_path.write_text("nope")
    with pytest.raises(UnsupportedFormatError):
        parse_document(bad_path)
