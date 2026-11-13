"""Turns the model's claimed source numbers into citations we can stand behind.

This module is the trust boundary. Everything above it treats Gemini's output
as a suggestion; everything below it is grounded in chunks the retriever
actually returned. Three things happen here, in order:

1. **Resolve.** A `source_id` of 3 becomes `chunks[2]`. Out-of-range and
   duplicate ids are dropped, and the dropped ones are counted rather than
   silently discarded — a model inventing source numbers is a real signal, and
   Day 15's telemetry should be able to see it.
2. **Enforce.** Under `require_citation_per_claim`, a claim left with zero
   valid citations invalidates the whole answer and the service abstains.
3. **Assemble.** Citations are built from our `RetrievedChunk` records, so
   filename, page, and chunk index come from PyMuPDF and the ingestion
   pipeline — never from the model's text.

**Why one bad claim abstains the whole answer rather than being dropped.**
Dropping it would ship a partial answer that reads as complete, and the reader
has no way to know a claim went missing. The Q5/Q6 promise is that every
displayed claim is verifiable and that the system refuses rather than guesses;
a silently truncated answer breaks both. It is the strict choice and it will
cost answer rate — that trade is measured by the Day 11 abstention metrics.
"""

from __future__ import annotations

from app.core.config import settings
from app.generation.schema import Answer, Citation, Claim, ModelAnswer
from app.retrieval.base import RetrievedChunk


def resolve_source_ids(
    source_ids: list[int], chunks: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], list[int]]:
    """Map 1-based ids to chunks. Returns (resolved, dropped) preserving order."""
    resolved: list[RetrievedChunk] = []
    dropped: list[int] = []
    seen: set[int] = set()
    for source_id in source_ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        if 1 <= source_id <= len(chunks):
            resolved.append(chunks[source_id - 1])
        else:
            dropped.append(source_id)
    return resolved, dropped


def assemble(
    model_answer: ModelAnswer,
    chunks: list[RetrievedChunk],
    *,
    require_citation_per_claim: bool | None = None,
) -> Answer:
    """Validate and ground the model's answer, or abstain."""
    strict = (
        settings.require_citation_per_claim
        if require_citation_per_claim is None
        else require_citation_per_claim
    )

    if not model_answer.sufficient_evidence:
        return Answer(
            answered=False,
            abstain_reason=model_answer.abstain_reason
            or "The retrieved sources do not answer this question.",
        )

    # The model can claim sufficient evidence and then produce nothing. Treat
    # that as the abstention it actually is rather than shipping an empty answer.
    if not model_answer.claims:
        return Answer(
            answered=False,
            abstain_reason="The model reported sufficient evidence but produced no claims.",
        )

    claims: list[Claim] = []
    all_dropped: list[int] = []
    for model_claim in model_answer.claims:
        resolved, dropped = resolve_source_ids(model_claim.source_ids, chunks)
        all_dropped.extend(dropped)

        if not resolved and strict:
            return Answer(
                answered=False,
                abstain_reason=(
                    "The answer contained a claim that could not be traced to a retrieved source."
                ),
                dropped_source_ids=tuple(all_dropped),
            )

        claims.append(
            Claim(
                text=model_claim.text,
                citations=tuple(Citation.from_chunk(chunk) for chunk in resolved),
            )
        )

    return Answer(answered=True, claims=tuple(claims), dropped_source_ids=tuple(all_dropped))
