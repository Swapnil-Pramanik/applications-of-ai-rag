"""Picks the hosted LLM from `settings.llm_provider` ("groq" or "gemini").

The only place that knows which provider is configured. Everything else asks
for a `StructuredClient` (answers) or a `GuideClient` (upload guides).
"""

from __future__ import annotations

from app.core.config import settings
from app.generation.gemini import GeminiClient, StructuredClient
from app.generation.groq import GroqClient
from app.generation.guide import GeminiGuideClient, GuideClient


def answer_client() -> StructuredClient:
    return GroqClient() if settings.llm_provider == "groq" else GeminiClient()


def guide_client() -> GuideClient:
    return GroqClient() if settings.llm_provider == "groq" else GeminiGuideClient()


def llm_label() -> str:
    """Human-readable name of the configured model, for the UI."""
    if settings.llm_provider == "groq":
        return f"{settings.groq_model.split('/')[-1]} on Groq"
    return f"{settings.gemini_model} (Google)"
