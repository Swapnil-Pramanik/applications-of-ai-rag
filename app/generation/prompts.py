"""The system instruction and the numbered SOURCES block.

**The prompt's whole job is to make abstention easy and citation cheap.** Left
to itself a model treats "I don't know" as a failure and will assemble an
answer out of whatever context it was handed. So the instruction states
abstention as the *expected* outcome for insufficient evidence, and the
structured schema gives it a boolean to set rather than a sentence to compose —
see `schema.ModelAnswer`.

**Sources are numbered, and the numbering is the contract.** The model cites
`[3]`; `citations.assemble` resolves 3 back to the chunk we put in position 3.
Filenames and page numbers appear in the block for the model's benefit — it
helps to know two chunks are from the same paper — but they are never read back
out of the model's output.
"""

from __future__ import annotations

from app.retrieval.base import RetrievedChunk

SYSTEM_INSTRUCTION = """\
You answer questions about machine-learning research papers using ONLY the numbered SOURCES \
provided in the user message.

Rules, in priority order:

1. If the SOURCES do not contain enough information to answer the question, set \
`sufficient_evidence` to false, leave `claims` empty, and say in one sentence what is missing. \
This is a correct and expected outcome, not a failure. Do not answer from your own knowledge of \
these papers, even when you are confident and even when the question is famous. A question you \
can answer from memory but not from the SOURCES is still an abstention.

2. When the SOURCES do support an answer, break it into individual claims. Each claim must be a \
single self-contained assertion of one or two sentences.

3. Every claim must list the source numbers that support it in `source_ids`. Cite the sources you \
actually used. Never cite a number that does not appear in the SOURCES block. A claim you cannot \
attribute to a specific source does not belong in the answer — drop it or abstain.

4. Prefer the wording of the source. Numbers, symbols, hyperparameters, and model names must be \
copied exactly as they appear; do not round, convert units, or normalise notation.

5. Answer the question that was asked. Do not add background the question did not request.\
"""


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """The numbered block the model cites into. 1-based to match `source_ids`."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{number}] {chunk.source_filename}, page {chunk.page_number}\n{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """The user turn. The question is repeated after the sources deliberately.

    Long-context models attend more reliably to the start and end of a prompt
    than to its middle, and the SOURCES block is by far the largest part of
    this one. Restating the question after it costs a handful of tokens and
    keeps the actual task adjacent to the generation point.
    """
    if not chunks:
        # Not an error: retrieval returning nothing is a legitimate state, and
        # the model should abstain on it rather than be asked to improvise.
        return (
            f"QUESTION: {question}\n\nSOURCES:\n(none — no documents matched this question)\n\n"
            "No sources were retrieved, so there is no evidence to answer from."
        )
    return (
        f"QUESTION: {question}\n\n"
        f"SOURCES:\n{format_sources(chunks)}\n\n"
        f"Answer this question using only the sources above: {question}"
    )
