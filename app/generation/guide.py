"""Source guide: one Gemini call per uploaded file, for the notebook UI.

Reads the start of a document and returns a short summary plus one question
for each demo case study, so the abstention behaviour can be shown on *any*
uploaded file, not just the 18 papers it was calibrated on:

| case | what the system should do |
|---|---|
| fact | answer from a single passage, with a citation |
| comparison | answer by combining two parts of the document |
| multi_hop | needs two separate passages in the top 5; may fail (the known weak spot) |
| not_in_document | sounds on-topic but the document never says; the *model* should refuse |

(The off-topic case needs no LLM: any question unrelated to the file works,
and the *gate* should refuse it before Gemini is called.)

The questions are suggestions written by an LLM, not verified labels: a
"not in document" question can turn out to be answerable. They are for
demonstrating behaviour live, never for computing a metric.
"""

from __future__ import annotations

from typing import Protocol

from google.genai import errors, types
from pydantic import BaseModel, Field

from app.generation.gemini import GeminiClient, GenerationUnavailableError

# Roughly 15k tokens: enough to see what a document is about, small enough that
# one upload is a cheap free-tier call.
MAX_GUIDE_CHARS = 60_000

GUIDE_INSTRUCTION = """\
You prepare a reading guide for a document a user has uploaded to a question-answering app.
The app answers ONLY from the document. Using only the document text you are given, write:

- summary: 2-3 plain sentences saying what the document is about.
- fact_question: a short question answered by ONE specific sentence or number in the document.
- comparison_question: a question that compares or contrasts two things the document describes.
- multi_hop_question: a question that needs two facts from two DIFFERENT parts of the
  document to be combined.
- not_in_document_question: a question that sounds natural for this document's topic, but
  which the document does NOT answer anywhere (e.g. a detail it never reports). Check carefully
  that the answer is absent.

Questions must be self-contained, under 25 words, and must not mention "the document".
"""


class SourceGuide(BaseModel):
    summary: str = Field(description="2-3 sentence summary of the document.")
    fact_question: str
    comparison_question: str
    multi_hop_question: str
    not_in_document_question: str


class GuideClient(Protocol):
    def generate_guide(self, text: str) -> SourceGuide: ...


class GeminiGuideClient(GeminiClient):
    """Same client, timeouts and retry policy as answering; different schema and prompt."""

    def _guide_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            http_options=self._http_options(),
            system_instruction=GUIDE_INSTRUCTION,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_schema=SourceGuide,
            temperature=0.2,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        )

    def generate_guide(self, text: str) -> SourceGuide:
        try:
            response = self.client.models.generate_content(
                model=self._model,
                contents=f"DOCUMENT:\n{text[:MAX_GUIDE_CHARS]}",
                config=self._guide_config(),
            )
        except errors.APIError as exc:
            raise GenerationUnavailableError(
                f"Gemini request failed ({exc.code}): {exc.message}", status_code=exc.code
            ) from exc
        if not isinstance(response.parsed, SourceGuide):
            raise GenerationUnavailableError("Gemini did not return a well-formed guide.")
        return response.parsed
