"""Loading and span-matching for the golden evaluation set.

**Why spans instead of chunk IDs.** A golden-set entry says "the answer to this
question is supported by *this exact sentence*, on *this page*, of *this PDF*" —
never "chunk 41". Chunk IDs are produced by ingestion, so they change meaning
the moment `chunk_size` or `chunk_overlap` changes, and Day 9's whole job is to
sweep exactly those parameters. Spans are a property of the document, so one
label set scores every configuration in the sweep fairly.

**Matching is normalized, not literal.** Spans are copied out of PyMuPDF's
extraction, which wraps lines mid-sentence, hyphenates across line breaks
(`Param-\\neters=110M`) and emits ligatures (`ﬁ`). The chunker then re-joins
atoms with single spaces. `normalize()` folds all of that away and is applied
to *both* sides of every comparison, so consistency is what matters, not
whether any individual transformation is linguistically correct.

**Keep spans short.** A span that straddles a chunk boundary is in neither
chunk and the question scores as a miss even when retrieval did its job. Spans
of roughly 5-15 words sit comfortably inside the 150-character overlap, so a
boundary can only ever cost one of the two neighbouring chunks, not both.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GOLDEN_SET_DIR = Path(__file__).resolve().parent.parent / "golden_set"
QUESTIONS_PATH = GOLDEN_SET_DIR / "questions.jsonl"

QuestionType = Literal["factual", "numeric", "comparative", "multi_hop"]
Relevance = Literal["any", "all"]

_LINE_BREAK_HYPHEN = re.compile(r"-\s+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold PDF-extraction noise so a span can be matched against page or chunk text.

    NFKC first (ligatures, compatibility forms), then join hyphenated line
    breaks, then collapse whitespace, then casefold.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_BREAK_HYPHEN.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().casefold()


@dataclass(frozen=True)
class SupportingSpan:
    source_filename: str  # e.g. "1706.03762.pdf" — matches the ingested filename
    page_number: int  # 1-indexed, as PyMuPDF reports it
    span: str  # verbatim from the extraction; whitespace already collapsed

    def is_supported_by(self, text: str) -> bool:
        return normalize(self.span) in normalize(text)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    answer: str
    type: QuestionType
    relevance: Relevance  # "any": one span suffices; "all": every span must be retrieved
    supporting_spans: tuple[SupportingSpan, ...]
    verified: bool  # flipped to true only by a human who checked it against the PDF
    notes: str = ""

    @property
    def source_filenames(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.source_filename for s in self.supporting_spans))

    def is_satisfied_by(self, texts: list[str]) -> bool:
        """Did this set of retrieved chunk texts surface the required support?"""
        hits = [any(span.is_supported_by(text) for text in texts) for span in self.supporting_spans]
        return all(hits) if self.relevance == "all" else any(hits)


def _parse(raw: dict[str, object]) -> GoldenQuestion:
    spans = raw["supporting_spans"]
    assert isinstance(spans, list)
    return GoldenQuestion(
        id=str(raw["id"]),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        type=raw["type"],  # type: ignore[arg-type]  # validated by validate_golden.py
        relevance=raw["relevance"],  # type: ignore[arg-type]
        supporting_spans=tuple(
            SupportingSpan(
                source_filename=str(s["source_filename"]),
                page_number=int(s["page_number"]),
                span=str(s["span"]),
            )
            for s in spans
        ),
        verified=bool(raw["verified"]),
        notes=str(raw.get("notes", "")),
    )


def load_golden_set(path: Path = QUESTIONS_PATH) -> list[GoldenQuestion]:
    with path.open(encoding="utf-8") as f:
        return [_parse(json.loads(line)) for line in f if line.strip()]
