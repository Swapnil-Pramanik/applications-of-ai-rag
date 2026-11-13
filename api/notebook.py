"""The notebook: uploaded sources, their guides, and source-scoped questions.

Wraps `DemoService` (retrieval, gate, Gemini answer: unchanged) and adds what a
NotebookLM-style app needs on top: per-source guides written in the
background, and removal. State is in memory; a restart empties the notebook.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.generation.gemini import GenerationUnavailableError
from app.generation.guide import GuideClient, SourceGuide
from app.generation.llm import guide_client
from app.ingestion.parsers import parse_document
from ui.service import DemoResult, DemoService, UploadResult

GuideStatus = Literal["pending", "ready", "failed"]

# Off-topic for any upload, so it needs no LLM: the gate should refuse it.
OFF_TOPIC_QUESTION = "What is the capital of France?"


@dataclass
class GuideState:
    status: GuideStatus = "pending"
    guide: SourceGuide | None = None
    error: str | None = None


@dataclass(frozen=True)
class CaseStudy:
    kind: str
    label: str
    question: str
    expect: str


@dataclass(frozen=True)
class SourceInfo:
    filename: str
    chunks: int
    guide: GuideState = field(default_factory=GuideState)

    @property
    def case_studies(self) -> list[CaseStudy]:
        g = self.guide.guide
        off_topic = CaseStudy(
            "off_topic", "Off-topic", OFF_TOPIC_QUESTION, "The gate refuses before calling the LLM"
        )
        if g is None:
            return [off_topic]
        return [
            CaseStudy("fact", "Fact", g.fact_question, "Answered with a citation"),
            CaseStudy("comparison", "Comparison", g.comparison_question, "Combines two passages"),
            CaseStudy(
                "multi_hop",
                "Multi-hop",
                g.multi_hop_question,
                "Needs two separate passages; may fail (known weak spot)",
            ),
            off_topic,
            CaseStudy(
                "not_in_document",
                "Not in document",
                g.not_in_document_question,
                "Refused: by the gate if nothing looks related, otherwise by the model",
            ),
        ]


Runner = Callable[[Callable[[], None]], None]


def _background(job: Callable[[], None]) -> None:
    threading.Thread(target=job, daemon=True).start()


class NotebookService:
    def __init__(
        self,
        demo: DemoService,
        *,
        guide_client: GuideClient | None = None,
        runner: Runner = _background,
    ):
        self._demo = demo
        self._guide_client = guide_client
        # Tests pass `lambda job: job()` to run the guide synchronously.
        self._run = runner
        self._guides: dict[str, GuideState] = {}

    @property
    def chunk_count(self) -> int:
        return self._demo.chunk_count

    def sources(self) -> list[SourceInfo]:
        counts = self._demo.chunk_counts()
        return [
            SourceInfo(name, counts.get(name, 0), self._guides.get(name, GuideState()))
            for name in self._demo.uploaded
        ]

    def add_source(self, path: Path) -> UploadResult:
        """Index the file now; write its guide in the background."""
        result = self._demo.add_document(path)
        # Read the text before returning: the caller deletes the temp file.
        text = "\n\n".join(page.text for page in parse_document(path))
        state = GuideState()
        self._guides[result.filename] = state

        def write_guide() -> None:
            try:
                client = self._guide_client or guide_client()
                state.guide = client.generate_guide(text)
                state.status = "ready"
            except GenerationUnavailableError as error:
                state.error = str(error)
                state.status = "failed"
            except Exception as error:  # a background thread must never leave "pending" forever
                state.error = f"could not write a guide: {error}"
                state.status = "failed"

        self._run(write_guide)
        return result

    def remove_source(self, filename: str) -> None:
        self._demo.remove_document(filename)
        self._guides.pop(filename, None)

    def ask(
        self, question: str, strategy: str, *, generate: bool, sources: Collection[str]
    ) -> DemoResult:
        return self._demo.ask(question, strategy, generate=generate, sources=sources)
