"""The Groq client and the provider switch, over a mock HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.core import config
from app.generation import llm
from app.generation.gemini import GeminiClient, GenerationUnavailableError
from app.generation.groq import GroqClient, inline_schema
from app.generation.guide import GeminiGuideClient
from app.generation.schema import ModelAnswer
from tests.unit.test_guide import GUIDE


def client_returning(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[GroqClient, list[dict[str, object]]]:
    sent: list[dict[str, object]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(record))
    return GroqClient(http, model="openai/gpt-oss-120b", api_key="test"), sent


def chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_an_answer_is_parsed_and_the_request_carries_the_schema() -> None:
    answer = ModelAnswer(sufficient_evidence=True, claims=[{"text": "h = 8.", "source_ids": [1]}])  # type: ignore[list-item]
    client, sent = client_returning(lambda r: chat(answer.model_dump_json()))
    assert client.generate("prompt") == answer
    body = sent[0]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["response_format"]["json_schema"]["name"] == "ModelAnswer"  # type: ignore[index]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]  # type: ignore[union-attr]


def test_malformed_output_is_an_abstention_not_an_exception() -> None:
    client, _ = client_returning(lambda r: chat("not json"))
    result = client.generate("prompt")
    assert not result.sufficient_evidence and "well-formed" in result.abstain_reason


def test_http_errors_and_timeouts_are_outages_with_status() -> None:
    quota = httpx.Response(429, json={"error": {"message": "rate limit reached"}})
    client, _ = client_returning(lambda r: quota)
    with pytest.raises(GenerationUnavailableError, match="rate limit") as caught:
        client.generate("prompt")
    assert caught.value.status_code == 429

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client, _ = client_returning(timeout)
    with pytest.raises(GenerationUnavailableError, match="Groq request failed"):
        client.generate("prompt")


def test_the_guide_is_parsed_and_truncated_and_a_bad_guide_is_an_outage() -> None:
    client, sent = client_returning(lambda r: chat(GUIDE.model_dump_json()))
    assert client.generate_guide("x" * 100_000) == GUIDE
    user = sent[0]["messages"][1]["content"]  # type: ignore[index]
    assert isinstance(user, str)
    assert len(user) <= config.settings.groq_guide_max_chars + len("DOCUMENT:\n")

    client, _ = client_returning(lambda r: chat("{}"))
    with pytest.raises(GenerationUnavailableError, match="well-formed guide"):
        client.generate_guide("text")


def test_a_missing_key_fails_only_when_used() -> None:
    client = GroqClient(api_key="")
    with pytest.raises(RuntimeError, match="GROQ"):
        client.generate("prompt")


def test_the_schema_has_no_refs() -> None:
    assert "$ref" not in json.dumps(inline_schema(ModelAnswer))


def test_the_provider_setting_picks_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "llm_provider", "groq")
    assert isinstance(llm.answer_client(), GroqClient)
    assert isinstance(llm.guide_client(), GroqClient)
    assert llm.llm_label() == "gpt-oss-120b"
    monkeypatch.setattr(config.settings, "llm_provider", "gemini")
    assert isinstance(llm.answer_client(), GeminiClient)
    assert isinstance(llm.guide_client(), GeminiGuideClient)
    assert llm.llm_label() == config.settings.gemini_model


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="llm_provider"):
        config.Settings(llm_provider="openai")
