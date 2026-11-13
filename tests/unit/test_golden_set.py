"""Tests for golden-set loading, span normalization and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.golden_set import (
    GoldenQuestion,
    SupportingSpan,
    load_golden_set,
    normalize,
)
from eval.harness.validate_golden import _check_schema, _check_spans, main

HEADS_SPAN = "we employ h = 8 parallel attention layers"


def _span(text: str = HEADS_SPAN, **kwargs: object) -> SupportingSpan:
    defaults: dict[str, object] = {"source_filename": "1706.03762.pdf", "page_number": 5}
    defaults.update(kwargs)
    return SupportingSpan(span=text, **defaults)  # type: ignore[arg-type]


def _question(**kwargs: object) -> GoldenQuestion:
    defaults: dict[str, object] = {
        "id": "g001",
        "question": "how many heads?",
        "answer": "eight",
        "type": "numeric",
        "relevance": "any",
        "supporting_spans": (_span(),),
        "verified": False,
    }
    defaults.update(kwargs)
    return GoldenQuestion(**defaults)  # type: ignore[arg-type]


class TestNormalize:
    def test_collapses_the_line_wrapping_pymupdf_emits(self) -> None:
        assert normalize("We used\nwarmup_steps = 4000.") == "we used warmup_steps = 4000."

    def test_rejoins_words_hyphenated_across_a_line_break(self) -> None:
        # PyMuPDF gives "Param-\neters=110M"; the chunker later gives "Param- eters=110M".
        assert normalize("Total Param-\neters=110M") == normalize("Total Param- eters=110M")
        assert normalize("Total Param-\neters=110M") == "total parameters=110m"

    def test_folds_ligatures_so_extracted_and_typed_text_agree(self) -> None:
        assert normalize("signiﬁcant") == normalize("significant")

    def test_is_case_insensitive(self) -> None:
        assert normalize("BERTBASE") == normalize("bertbase")


class TestSupportingSpan:
    def test_matches_across_a_line_break(self) -> None:
        span = _span(HEADS_SPAN)
        page = "In this work we employ h = 8\nparallel attention layers, or heads."
        assert span.is_supported_by(page)

    def test_does_not_match_unrelated_text(self) -> None:
        other = "The decoder is also composed of a stack of N = 6 layers."
        assert not _span().is_supported_by(other)


class TestIsSatisfiedBy:
    def test_any_relevance_needs_one_span(self) -> None:
        question = _question(
            relevance="any",
            supporting_spans=(_span("h = 8 parallel"), _span("dmodel = 512")),
        )
        assert question.is_satisfied_by(["... h = 8 parallel attention layers ..."])

    def test_all_relevance_needs_every_span(self) -> None:
        question = _question(
            relevance="all",
            supporting_spans=(_span("h = 8 parallel"), _span("dmodel = 512")),
        )
        assert not question.is_satisfied_by(["... h = 8 parallel attention layers ..."])
        assert question.is_satisfied_by(["... h = 8 parallel ...", "... dmodel = 512 ..."])

    def test_spans_may_be_spread_over_different_chunks(self) -> None:
        question = _question(
            relevance="all",
            supporting_spans=(_span("h = 8"), _span("dmodel = 512")),
        )
        assert question.is_satisfied_by(["h = 8 heads", "outputs of dimension dmodel = 512"])


class TestLoad:
    def test_round_trips_a_written_file(self, tmp_path: Path) -> None:
        path = tmp_path / "questions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "g001",
                    "question": "q?",
                    "answer": "a",
                    "type": "factual",
                    "relevance": "any",
                    "supporting_spans": [
                        {"source_filename": "1706.03762.pdf", "page_number": 3, "span": "s"}
                    ],
                    "verified": True,
                    "notes": "checked",
                }
            )
            + "\n\n",  # trailing blank line is ignored
            encoding="utf-8",
        )
        (question,) = load_golden_set(path)
        assert question.verified is True
        assert question.notes == "checked"
        assert question.source_filenames == ("1706.03762.pdf",)

    def test_source_filenames_deduplicates_and_keeps_order(self) -> None:
        question = _question(
            relevance="all",
            supporting_spans=(
                _span(source_filename="b.pdf"),
                _span(source_filename="a.pdf"),
                _span(source_filename="b.pdf"),
            ),
        )
        assert question.source_filenames == ("b.pdf", "a.pdf")


class TestSchemaChecks:
    def test_accepts_the_committed_golden_set(self) -> None:
        assert _check_schema(load_golden_set()) == []

    def test_flags_duplicate_ids(self) -> None:
        errors = _check_schema([_question(), _question()])
        assert any("duplicate id" in e for e in errors)

    def test_flags_unknown_type_and_relevance(self) -> None:
        errors = _check_schema([_question(type="trivia", relevance="some")])
        assert any("unknown type" in e for e in errors)
        assert any("unknown relevance" in e for e in errors)

    def test_flags_a_paper_outside_the_corpus(self) -> None:
        spans = (_span(source_filename="9999.99999.pdf"),)
        errors = _check_schema([_question(supporting_spans=spans)])
        assert any("not in the corpus" in e for e in errors)

    def test_flags_an_entry_with_no_spans(self) -> None:
        errors = _check_schema([_question(supporting_spans=())])
        assert any("no supporting spans" in e for e in errors)

    def test_flags_all_relevance_with_a_single_span(self) -> None:
        errors = _check_schema([_question(relevance="all")])
        assert any("use 'any'" in e for e in errors)

    def test_flags_a_span_long_enough_to_straddle_a_chunk_boundary(self) -> None:
        errors = _check_schema([_question(supporting_spans=(_span("x" * 201),))])
        assert any("exceeds" in e for e in errors)


class TestSpanChecks:
    def test_reports_a_missing_pdf_rather_than_crashing(self, tmp_path: Path) -> None:
        errors = _check_spans([_question()], tmp_path)
        assert any("run `make corpus`" in e for e in errors)

    def test_schema_only_run_needs_no_corpus(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--schema-only"]) == 0
        assert "60 questions" in capsys.readouterr().out

    def test_reports_failures_and_exits_non_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "questions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "g001",
                    "question": "q?",
                    "answer": "a",
                    "type": "factual",
                    "relevance": "any",
                    "supporting_spans": [
                        {"source_filename": "not-a-paper.pdf", "page_number": 1, "span": "s"}
                    ],
                    "verified": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert main(["--questions", str(path), "--schema-only"]) == 1
