"""The shapes crossing the generation boundary — model output, and what we ship.

**Two different types on purpose.** `ModelAnswer` is what Gemini is asked to
produce and is *untrusted*: it is a language model's JSON, and the citation
fields in it are 1-based integers pointing into the numbered context block.
`Answer` is what leaves this package, and its citations are real
`RetrievedChunk` records the retriever actually returned.

**Why the model cites by index and never by filename or page.** Asking for
`"source": "1706.03762.pdf, p.7"` reads more natural and is unverifiable — the
model can emit a plausible filename and page for a chunk it never saw, and
nothing downstream can tell. An integer is checkable: it either indexes a chunk
we put in the prompt or it doesn't, and `citations.assemble` drops the ones that
don't. The citation is then constructed from *our* record of that chunk, so the
page number in the UI comes from PyMuPDF, not from the model.

This is the mechanism behind the Q5 promise that every claim maps to a source
doc + page + chunk ID that a reader can go verify.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.retrieval.base import RetrievedChunk


class ModelClaim(BaseModel):
    """One assertion the model makes, plus the sources it says support it."""

    text: str = Field(description="A single factual claim, stated in one or two sentences.")
    source_ids: list[int] = Field(
        default_factory=list,
        description="1-based numbers of the sources supporting this claim, from the SOURCES block.",
    )


class ModelAnswer(BaseModel):
    """The structured output schema handed to Gemini as `response_schema`.

    `sufficient_evidence` is the model-side half of the Q6 abstention policy.
    It is a separate boolean rather than a magic string in `answer` because a
    sentinel the model has to spell correctly is a sentinel it will eventually
    misspell — and a missed abstention is a hallucination shipped with a
    citation attached to it.
    """

    sufficient_evidence: bool = Field(
        description="True only if the SOURCES below actually answer the question."
    )
    claims: list[ModelClaim] = Field(
        default_factory=list,
        description="The answer, decomposed into individually-cited claims. Empty when abstaining.",
    )
    abstain_reason: str = Field(
        default="",
        description="When sufficient_evidence is false, one sentence on what is missing.",
    )


@dataclass(frozen=True)
class Citation:
    """A resolved pointer into the corpus. Every field comes from our records."""

    source_filename: str
    page_number: int
    chunk_index: int
    vector_id: int
    quote: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, *, quote_chars: int = 300) -> Citation:
        quote = chunk.text.strip()
        if len(quote) > quote_chars:
            quote = quote[:quote_chars].rstrip() + "…"
        return cls(
            source_filename=chunk.source_filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            vector_id=chunk.vector_id,
            quote=quote,
        )


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Answer:
    """What the API returns. `answered=False` is a first-class outcome, not an error."""

    answered: bool
    claims: tuple[Claim, ...] = ()
    abstain_reason: str = ""
    # Kept for observability rather than display: a nonzero count means the
    # model invented source numbers, which is worth a metric and an alert
    # long before it is worth showing a user.
    dropped_source_ids: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(claim.text for claim in self.claims)

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Every citation across all claims, de-duplicated, in first-use order."""
        seen: dict[tuple[str, int, int], Citation] = {}
        for claim in self.claims:
            for citation in claim.citations:
                key = (citation.source_filename, citation.page_number, citation.chunk_index)
                seen.setdefault(key, citation)
        return tuple(seen.values())
