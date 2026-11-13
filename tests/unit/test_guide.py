"""The source guide call, with a fake SDK client (no network)."""

from __future__ import annotations

import pytest
from google.genai import errors

from app.generation.gemini import GenerationUnavailableError
from app.generation.guide import MAX_GUIDE_CHARS, GeminiGuideClient, SourceGuide

GUIDE = SourceGuide(
    summary="About zebrafish.",
    fact_question="What learning rate is used?",
    comparison_question="How do A and B differ?",
    multi_hop_question="Which X and what is its Y?",
    not_in_document_question="What did it cost?",
)


class Parsed:
    def __init__(self, parsed: object):
        self.parsed = parsed


class FakeModels:
    def __init__(self, parsed: object):
        self._parsed = parsed
        self.kwargs: dict[str, object] = {}

    def generate_content(self, **kwargs: object) -> Parsed:
        self.kwargs = kwargs
        return Parsed(self._parsed)


class FakeSDKClient:
    def __init__(self, parsed: object):
        self.models = FakeModels(parsed)


def test_the_guide_is_parsed_and_the_document_is_truncated() -> None:
    sdk = FakeSDKClient(GUIDE)
    guide = GeminiGuideClient(client=sdk).generate_guide("x" * (MAX_GUIDE_CHARS + 500))  # type: ignore[arg-type]
    assert guide == GUIDE
    contents = sdk.models.kwargs["contents"]
    assert isinstance(contents, str) and len(contents) <= MAX_GUIDE_CHARS + len("DOCUMENT:\n")


def test_a_malformed_guide_is_an_outage() -> None:
    with pytest.raises(GenerationUnavailableError, match="well-formed"):
        GeminiGuideClient(client=FakeSDKClient(None)).generate_guide("text")  # type: ignore[arg-type]


def test_an_api_error_is_an_outage_with_its_status() -> None:
    class Failing:
        def generate_content(self, **kwargs: object) -> None:
            raise errors.ClientError(
                429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
            )

    class SDK:
        models = Failing()

    with pytest.raises(GenerationUnavailableError) as caught:
        GeminiGuideClient(client=SDK()).generate_guide("text")  # type: ignore[arg-type]
    assert caught.value.status_code == 429
