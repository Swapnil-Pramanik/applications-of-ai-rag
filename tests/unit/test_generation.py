"""Generation tests — the trust boundary, not the SDK.

Nothing here touches the network. What is worth testing is every way the model
can hand back something we must not ship: invented source numbers, claims with
no citations at all, a claimed answer with no content, and unparseable output.
Each of those produces a plausible-looking wrong answer if unguarded, which is
the failure class this project exists to avoid.
"""

from __future__ import annotations

import pytest

from app.generation.citations import assemble, resolve_source_ids
from app.generation.gemini import generate_answer
from app.generation.prompts import build_prompt, format_sources
from app.generation.schema import Answer, ModelAnswer, ModelClaim
from app.retrieval.base import RetrievedChunk


def chunk(vector_id: int, text: str, *, page: int = 1, source: str = "1706.03762.pdf"):
    return RetrievedChunk(
        vector_id=vector_id,
        text=text,
        page_number=page,
        chunk_index=vector_id,
        source_filename=source,
        score=1.0 / (vector_id + 1),
    )


CHUNKS = [
    chunk(0, "We used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98.", page=7),
    chunk(1, "The base model was trained for 100,000 steps.", page=7),
    chunk(2, "Adam's recommended default is beta_2 = 0.999.", page=2, source="1412.6980.pdf"),
]


class StubClient:
    def __init__(self, answer: ModelAnswer):
        self._answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ModelAnswer:
        self.prompts.append(prompt)
        return self._answer


# --- the numbered-source contract -----------------------------------------


def test_sources_are_numbered_from_one_and_carry_filename_and_page() -> None:
    block = format_sources(CHUNKS)
    assert block.startswith("[1] 1706.03762.pdf, page 7")
    assert "[3] 1412.6980.pdf, page 2" in block


def test_prompt_restates_the_question_after_the_sources() -> None:
    """Long prompts attend better at the ends; the sources block is the middle."""
    prompt = build_prompt("Which optimizer?", CHUNKS)
    assert prompt.index("Which optimizer?") < prompt.index("SOURCES:")
    assert prompt.rindex("Which optimizer?") > prompt.index("SOURCES:")


def test_no_retrieved_chunks_is_a_prompt_not_an_error() -> None:
    prompt = build_prompt("Anything?", [])
    assert "(none" in prompt


# --- resolution ------------------------------------------------------------


def test_source_ids_are_one_based() -> None:
    resolved, dropped = resolve_source_ids([1, 3], CHUNKS)
    assert [c.vector_id for c in resolved] == [0, 2]
    assert dropped == []


@pytest.mark.parametrize("bad_id", [0, 4, -1, 99])
def test_out_of_range_source_ids_are_dropped_and_counted(bad_id: int) -> None:
    """A model inventing source numbers must never reach a citation."""
    resolved, dropped = resolve_source_ids([1, bad_id], CHUNKS)
    assert [c.vector_id for c in resolved] == [0]
    assert dropped == [bad_id]


def test_duplicate_source_ids_cite_once() -> None:
    resolved, dropped = resolve_source_ids([2, 2, 2], CHUNKS)
    assert len(resolved) == 1 and dropped == []


# --- assembly and the abstention policy ------------------------------------


def test_a_supported_answer_carries_citations_built_from_our_records() -> None:
    model_answer = ModelAnswer(
        sufficient_evidence=True,
        claims=[ModelClaim(text="They used Adam with beta_2 = 0.98.", source_ids=[1])],
    )
    answer = assemble(model_answer, CHUNKS)

    assert answer.answered
    citation = answer.claims[0].citations[0]
    # Page and filename come from ingestion, never from the model's text.
    assert (citation.source_filename, citation.page_number) == ("1706.03762.pdf", 7)
    assert citation.vector_id == 0


def test_the_model_abstaining_is_passed_through() -> None:
    answer = assemble(
        ModelAnswer(sufficient_evidence=False, abstain_reason="No source discusses this."), CHUNKS
    )
    assert not answer.answered
    assert answer.abstain_reason == "No source discusses this."


def test_claiming_evidence_while_producing_nothing_is_an_abstention() -> None:
    answer = assemble(ModelAnswer(sufficient_evidence=True, claims=[]), CHUNKS)
    assert not answer.answered


def test_an_uncited_claim_abstains_the_whole_answer_under_the_strict_policy() -> None:
    """A partial answer reads as complete — the reader cannot see what was dropped."""
    model_answer = ModelAnswer(
        sufficient_evidence=True,
        claims=[
            ModelClaim(text="They used Adam.", source_ids=[1]),
            ModelClaim(text="It was trained on 8 GPUs.", source_ids=[9]),
        ],
    )
    answer = assemble(model_answer, CHUNKS, require_citation_per_claim=True)

    assert not answer.answered
    assert answer.dropped_source_ids == (9,)


def test_the_lenient_policy_keeps_the_claim_but_still_records_the_drop() -> None:
    model_answer = ModelAnswer(
        sufficient_evidence=True,
        claims=[ModelClaim(text="It was trained on 8 GPUs.", source_ids=[9])],
    )
    answer = assemble(model_answer, CHUNKS, require_citation_per_claim=False)

    assert answer.answered
    assert answer.claims[0].citations == ()
    assert answer.dropped_source_ids == (9,)


def test_citations_are_deduplicated_across_claims_in_first_use_order() -> None:
    model_answer = ModelAnswer(
        sufficient_evidence=True,
        claims=[
            ModelClaim(text="Adam, beta_2 = 0.98.", source_ids=[1]),
            ModelClaim(text="Adam's default is 0.999.", source_ids=[3, 1]),
        ],
    )
    answer = assemble(model_answer, CHUNKS)
    assert [c.vector_id for c in answer.citations] == [0, 2]


def test_a_long_chunk_is_truncated_in_the_quote_but_not_in_the_record() -> None:
    long_chunk = chunk(0, "x" * 500)
    answer = assemble(
        ModelAnswer(sufficient_evidence=True, claims=[ModelClaim(text="A.", source_ids=[1])]),
        [long_chunk],
    )
    quote = answer.claims[0].citations[0].quote
    assert quote.endswith("…") and len(quote) <= 301


# --- end to end through a stub --------------------------------------------


def test_generate_answer_preserves_retrieval_order_into_the_numbering() -> None:
    """Source [1] must be the reranker's best chunk — a reorder repoints every citation."""
    stub = StubClient(
        ModelAnswer(
            sufficient_evidence=True,
            claims=[ModelClaim(text="beta_2 = 0.98.", source_ids=[1])],
        )
    )
    answer = generate_answer("Which optimizer?", CHUNKS, client=stub)

    assert answer.answered
    assert answer.claims[0].citations[0].vector_id == CHUNKS[0].vector_id
    assert "[1] 1706.03762.pdf" in stub.prompts[0]


def test_answer_text_joins_claims() -> None:
    stub = StubClient(
        ModelAnswer(
            sufficient_evidence=True,
            claims=[
                ModelClaim(text="They used Adam.", source_ids=[1]),
                ModelClaim(text="Beta_2 was 0.98.", source_ids=[1]),
            ],
        )
    )
    answer = generate_answer("q", CHUNKS, client=stub)
    assert answer.text == "They used Adam. Beta_2 was 0.98."


def test_empty_answer_has_no_text_and_no_citations() -> None:
    assert Answer(answered=False).text == ""
    assert Answer(answered=False).citations == ()


# --- the client wrapper, without the network ------------------------------


def test_unparseable_model_output_becomes_an_abstention_not_an_exception() -> None:
    """`parsed` comes back None when the model is cut off by max_output_tokens.

    Truncated output is precisely when improvising is worst, so the correct
    response is to abstain rather than raise or salvage.
    """

    class Truncated:
        parsed = None

    class FakeModels:
        def generate_content(self, **kwargs: object) -> Truncated:
            return Truncated()

    class FakeSDKClient:
        models = FakeModels()

    from app.generation.gemini import GeminiClient

    model_answer = GeminiClient(client=FakeSDKClient()).generate("prompt")  # type: ignore[arg-type]
    assert not model_answer.sufficient_evidence
    assert "well-formed" in model_answer.abstain_reason


def test_building_a_client_without_an_api_key_fails_loudly_but_only_when_used() -> None:
    """Importing generation must not require credentials — same lesson as db/session."""
    from app.core import config
    from app.generation.gemini import GeminiClient

    original = config.settings.gemini_api_key
    config.settings.gemini_api_key = ""
    try:
        client = GeminiClient()  # constructing is fine
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            _ = client.client
    finally:
        config.settings.gemini_api_key = original


def test_an_upstream_failure_is_an_outage_not_an_abstention() -> None:
    """503 must not inflate the abstention rate — that number is a product claim."""
    from google.genai import errors

    from app.generation.gemini import GeminiClient, GenerationUnavailableError

    class Failing:
        def generate_content(self, **kwargs: object) -> None:
            raise errors.ServerError(
                503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
            )

    class FakeSDKClient:
        models = Failing()

    with pytest.raises(GenerationUnavailableError, match="503") as caught:
        GeminiClient(client=FakeSDKClient()).generate("prompt")  # type: ignore[arg-type]
    # 503 is worth retrying; 429 (quota spent) is not. Day 13 needs to tell them apart.
    assert caught.value.status_code == 503


def test_the_request_is_bounded_by_a_timeout_and_capped_retries() -> None:
    """An unbounded SDK retry turned one 503 into a 128s hang while measuring Day 10."""
    from app.core.config import settings
    from app.generation.gemini import GeminiClient

    options = GeminiClient(client=object())._http_options()  # type: ignore[arg-type]
    assert options.timeout == int(settings.gemini_timeout_seconds * 1000)
    assert options.retry_options is not None
    assert options.retry_options.attempts == settings.gemini_retry_attempts
