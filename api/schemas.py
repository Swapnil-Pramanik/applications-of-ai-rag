"""JSON shapes the API returns, and the one function that builds them.

Everything the frontend needs is computed here (paper titles, footnote
numbers, which passages passed the gate), so the frontend only draws, the same
split as `ui/service.py` and `ui/streamlit_app.py`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ui.service import STRATEGY_LABELS, DemoResult, Status, numbered_claims, paper_title


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    strategy: str = "hybrid+rerank"
    generate: bool = True


class Passage(BaseModel):
    rank: int
    title: str
    source_filename: str
    page_number: int
    score: float
    passed_gate: bool | None
    text: str


class Source(BaseModel):
    number: int
    title: str
    source_filename: str
    page_number: int
    quote: str


class ClaimOut(BaseModel):
    text: str
    sources: list[int]


class GateOut(BaseModel):
    passed: bool
    threshold: float
    top_score: float | None
    reason: str


class QueryResponse(BaseModel):
    question: str
    strategy: str
    strategy_label: str
    status: Status
    claims: list[ClaimOut]
    sources: list[Source]
    abstain_reason: str | None
    error: str | None
    gate: GateOut | None
    passages: list[Passage]
    seconds: float


def to_response(result: DemoResult) -> QueryResponse:
    answer = result.answer
    answered = answer is not None and answer.answered
    return QueryResponse(
        question=result.question,
        strategy=result.strategy,
        strategy_label=STRATEGY_LABELS[result.strategy],
        status=result.status,
        claims=[
            ClaimOut(text=claim.text, sources=numbers)
            for claim, numbers in (numbered_claims(answer) if answered and answer else [])
        ],
        sources=[
            Source(
                number=n,
                title=paper_title(c.source_filename),
                source_filename=c.source_filename,
                page_number=c.page_number,
                quote=c.quote,
            )
            for n, c in enumerate(answer.citations if answered and answer else (), start=1)
        ],
        abstain_reason=answer.abstain_reason if answer is not None and not answered else None,
        error=result.error,
        gate=(
            GateOut(
                passed=result.gate.passed,
                threshold=result.gate.threshold,
                top_score=result.gate.top_score,
                reason=result.gate.reason,
            )
            if result.gate is not None
            else None
        ),
        passages=[
            Passage(
                rank=rank,
                title=paper_title(chunk.source_filename),
                source_filename=chunk.source_filename,
                page_number=chunk.page_number,
                score=chunk.score,
                passed_gate=result.passed_gate(chunk),
                text=chunk.text,
            )
            for rank, chunk in enumerate(result.retrieved, start=1)
        ],
        seconds=round(result.seconds, 3),
    )


class Paper(BaseModel):
    title: str
    chunks: int
    uploaded: bool


class StrategyOut(BaseModel):
    key: str
    label: str


class ExampleOut(BaseModel):
    label: str
    question: str


class MetaResponse(BaseModel):
    papers: list[Paper]
    chunk_count: int
    strategies: list[StrategyOut]
    examples: list[ExampleOut]


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    chunk_count: int
