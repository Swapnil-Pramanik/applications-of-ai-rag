"""The Gemini client, and the end-to-end generate step.

**Everything that can change an answer is config, not a literal.** Model
version, thinking level, temperature and output cap all live in
`app/core/config.py`, because Day 12's RAGAS judge and Day 17's latency work
both need to vary them, and because a committed faithfulness number is only
reproducible if the configuration that produced it is recorded.

**Pinned model, never `gemini-flash-latest`.** A floating alias silently
changes the model under committed eval numbers.

**`thinking_level` is set deliberately.** `gemini-3.6-flash` does
chain-of-thought by default; measured on a realistic 5-chunk RAG prompt that is
4.14s mean against 2.49s at `LOW`. Default thinking exceeds the 3.0s
end-to-end reference point on its own, before retrieval is counted.

**The client is constructed lazily and injectable.** Importing this module must
not require an API key — the same lesson as `db/session.py`, where an
import-time engine broke collection for every test in the suite. Tests pass a
stub; nothing here touches the network unless a real client is built.
"""

from __future__ import annotations

from typing import Protocol

from google import genai
from google.genai import errors, types

from app.core.config import settings
from app.generation.citations import assemble
from app.generation.prompts import SYSTEM_INSTRUCTION, build_prompt
from app.generation.schema import Answer, ModelAnswer
from app.retrieval.base import RetrievedChunk


class GenerationUnavailableError(RuntimeError):
    """The upstream model could not be reached or failed.

    **Deliberately not an abstention.** Abstaining means "the corpus does not
    support an answer" and is a product decision we measure (Q6, and the Day 11
    correct-abstention metric). A 503 means "Gemini is busy" and is an outage.
    Folding one into the other would quietly inflate the abstention rate every
    time the upstream had a bad afternoon, corrupting the exact number this
    project uses to argue it does not hallucinate. The API layer maps this to
    an error response, not to a normal answer payload.

    `status_code` is carried through because the two failures we actually
    measured want different handling: 503 ("high demand") is transient and
    worth retrying, while 429 means the free-tier quota is spent and retrying
    only makes it worse. Day 13 needs that distinction to choose a response.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class StructuredClient(Protocol):
    """The one call generation needs. Narrow on purpose so a stub is three lines."""

    def generate(self, prompt: str) -> ModelAnswer: ...


class GeminiClient:
    """Wraps `google-genai` and returns a parsed `ModelAnswer`."""

    def __init__(self, client: genai.Client | None = None, *, model: str | None = None):
        self._client = client
        self._model = model or settings.gemini_model

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set; cannot build a Gemini client")
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def _http_options(self) -> types.HttpOptions:
        return types.HttpOptions(
            timeout=int(settings.gemini_timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=settings.gemini_retry_attempts),
        )

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            http_options=self._http_options(),
            system_instruction=SYSTEM_INSTRUCTION,
            # The SDK reads a Pydantic `response_schema` as a possible tool and
            # warns about automatic function calling on every request. We have
            # no tools; disabling it silences the warning and removes a code
            # path that could otherwise turn one call into several.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_schema=ModelAnswer,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            thinking_config=types.ThinkingConfig(
                # Config carries a plain string so `app/core/config.py` stays
                # free of SDK imports. The value is validated there, which
                # matters: `ThinkingLevel("lo")` does not raise, it warns and
                # returns a synthetic member that reaches the API unnoticed.
                thinking_level=types.ThinkingLevel(settings.gemini_thinking_level),
            ),
        )

    def generate(self, prompt: str) -> ModelAnswer:
        try:
            response = self.client.models.generate_content(
                model=self._model, contents=prompt, config=self._config()
            )
        except errors.APIError as exc:
            # Both of these were hit for real while measuring Day 10: 503
            # "high demand" on 5 of 10 calls, then 429 once the free-tier
            # quota was spent. Surface them as an outage rather than letting
            # them 500, and keep them out of the abstention numbers.
            raise GenerationUnavailableError(
                f"Gemini request failed ({exc.code}): {exc.message}", status_code=exc.code
            ) from exc
        parsed = response.parsed
        # `response_schema` makes this the overwhelmingly common path, but the
        # SDK still types `parsed` as optional and it comes back None when the
        # model hits `max_output_tokens` mid-object. Abstaining is the correct
        # response to unparseable output: the alternative is raising, and a
        # truncated answer is exactly when we least want to improvise.
        if not isinstance(parsed, ModelAnswer):
            return ModelAnswer(
                sufficient_evidence=False,
                abstain_reason="The model did not return a well-formed answer.",
            )
        return parsed


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    client: StructuredClient | None = None,
) -> Answer:
    """Retrieved chunks in, grounded and citation-checked answer out.

    Retrieval order is preserved into the prompt: the numbering the model cites
    is the reranker's ranking, so source [1] is the best chunk. Nothing here
    re-sorts, because `assemble` resolves ids positionally and a reorder would
    silently repoint every citation.
    """
    client = client or GeminiClient()
    model_answer = client.generate(build_prompt(question, chunks))
    return assemble(model_answer, chunks)
