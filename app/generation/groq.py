"""Groq client: the open-weight `gpt-oss-120b` behind the same two interfaces as Gemini.

Groq serves open-weight models on a free tier with far more headroom than
Gemini's (about 1,000 requests/day for gpt-oss-120b, against Gemini's 20),
through an OpenAI-compatible HTTP API. Called with plain `httpx` rather than
another SDK: one POST is all either interface needs.

It implements both `StructuredClient.generate` (cited answers) and
`GuideClient.generate_guide` (per-upload guides), with the same prompts and
the same Pydantic schemas as the Gemini path, so retrieval, the gate and
citation checking cannot tell which provider answered.

**Free-tier budget:** 8,000 tokens per minute for gpt-oss-120b, counting the
prompt and the reserved reply. A guide reads only the first
`groq_guide_max_chars` of a document, and reply budgets are capped (see
`app/core/config.py`). A 413/429 from Groq is reported as an outage.

Failure policy matches `gemini.py`: an HTTP error or timeout is an outage
(`GenerationUnavailableError`), never an abstention; an answer that does not
parse into the schema becomes an abstention, because improvising around
malformed output is exactly how a citation gets invented.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.generation.gemini import GenerationUnavailableError
from app.generation.guide import GUIDE_INSTRUCTION, SourceGuide
from app.generation.prompts import SYSTEM_INSTRUCTION
from app.generation.schema import ModelAnswer

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema with `$ref`s expanded, so any provider can read it."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def expand(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return expand(defs[node["$ref"].split("/")[-1]])
            return {k: expand(v) for k, v in node.items()}
        if isinstance(node, list):
            return [expand(v) for v in node]
        return node

    expanded: dict[str, Any] = expand(schema)
    return expanded


class GroqClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self._http = http
        self._model = model or settings.groq_model
        self._api_key = api_key if api_key is not None else settings.groq_api_key

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            if not self._api_key:
                raise RuntimeError("GROQ_API_KEY (or GROQ_KEY) is not set; cannot call Groq")
            self._http = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=settings.groq_timeout_seconds,
            )
        return self._http

    def _complete(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            # gpt-oss reasons before answering; "low" keeps latency and token
            # use down, the same trade made with Gemini's thinking level.
            "reasoning_effort": "low",
            "max_completion_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": inline_schema(schema)},
            },
        }
        try:
            response = self.http.post(GROQ_URL, json=body)
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError(f"Groq request failed: {exc}") from exc
        if response.status_code != 200:
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text[:200]
            raise GenerationUnavailableError(
                f"Groq request failed ({response.status_code}): {message}",
                status_code=response.status_code,
            )
        try:
            content: str = response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            content = ""
        return content

    def generate(self, prompt: str) -> ModelAnswer:
        content = self._complete(
            SYSTEM_INSTRUCTION,
            prompt,
            ModelAnswer,
            temperature=settings.gemini_temperature,
            max_tokens=settings.groq_answer_max_tokens,
        )
        try:
            return ModelAnswer.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError):
            return ModelAnswer(
                sufficient_evidence=False,
                abstain_reason="The model did not return a well-formed answer.",
            )

    def generate_guide(self, text: str) -> SourceGuide:
        content = self._complete(
            GUIDE_INSTRUCTION,
            f"DOCUMENT:\n{text[: settings.groq_guide_max_chars]}",
            SourceGuide,
            temperature=0.2,
            max_tokens=settings.groq_guide_max_tokens,
        )
        try:
            return SourceGuide.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise GenerationUnavailableError("Groq did not return a well-formed guide.") from exc
