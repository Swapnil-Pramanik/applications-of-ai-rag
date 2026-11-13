"""The notebook HTTP API, with stub models, LLM and guide writer."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.notebook import OFF_TOPIC_QUESTION, NotebookService
from app.generation.gemini import GenerationUnavailableError
from app.generation.guide import SourceGuide
from app.ingestion.embedder import Embedder
from eval.harness.corpus_index import EvalCorpus
from tests.unit.test_guide import GUIDE
from tests.unit.test_ui_service import FixedReranker, OnesModel, StubClient
from ui.service import DemoService

# Several chunks, with "zebrafish" in only one: BM25's IDF needs a few chunks to
# score anything (on a 1-2 chunk document every term's IDF is <= 0).
FILLER = "".join(
    f"Paragraph {i} discusses unrelated topic number {i} at length. " * 6 + "\n\n" for i in range(6)
)
NOTES = (
    "notes.txt",
    (FILLER + "The zebrafish optimizer uses a learning rate of 0.42.").encode(),
    "text/plain",
)


class StubGuides:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.texts: list[str] = []

    def generate_guide(self, text: str) -> SourceGuide:
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        return GUIDE


def notebook(
    score: float = 5.0, client: StubClient | None = None, guides: StubGuides | None = None
) -> NotebookService:
    demo = DemoService(
        EvalCorpus.empty(),
        reranker=FixedReranker(score),
        client=client or StubClient(),
        embedder=Embedder(model=OnesModel()),  # type: ignore[arg-type]
    )
    return NotebookService(demo, guide_client=guides or StubGuides(), runner=lambda job: job())


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(notebook())) as c:
        yield c


def add(c: TestClient, file: tuple[str, bytes, str] = NOTES) -> dict[str, object]:
    response = c.post("/api/sources", files={"file": file})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def ask(c: TestClient, question: str, **extra: object) -> dict[str, object]:
    body = {"question": question, "sources": ["notes.txt"], **extra}
    response = c.post("/api/query", json=body)
    assert response.status_code == 200, response.text
    data: dict[str, object] = response.json()
    return data


def test_the_notebook_starts_empty(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "sources": 0, "chunks": 0}
    assert client.get("/api/sources").json() == []
    meta = client.get("/api/meta").json()
    assert meta["gate_threshold"] == -2.0 and {s["key"] for s in meta["strategies"]} >= {
        "hybrid+rerank"
    }


def test_asking_before_adding_a_source_is_rejected(client: TestClient) -> None:
    r = client.post("/api/query", json={"question": "anything", "sources": ["notes.txt"]})
    assert r.status_code == 422 and "no sources" in r.json()["detail"]
    assert client.post("/api/query", json={"question": "x", "sources": []}).status_code == 422


def test_an_added_source_gets_a_guide_with_five_case_studies(client: TestClient) -> None:
    source = add(client)
    assert source["filename"] == "notes.txt" and source["guide_status"] == "ready"
    assert source["summary"] == GUIDE.summary
    cases = {c["kind"]: c["question"] for c in source["case_studies"]}  # type: ignore[attr-defined]
    assert cases == {
        "fact": GUIDE.fact_question,
        "comparison": GUIDE.comparison_question,
        "multi_hop": GUIDE.multi_hop_question,
        "off_topic": OFF_TOPIC_QUESTION,
        "not_in_document": GUIDE.not_in_document_question,
    }
    assert client.get("/api/sources").json() == [source]


def test_a_failed_guide_still_leaves_the_source_usable() -> None:
    guides = StubGuides(error=GenerationUnavailableError("busy", status_code=503))
    with TestClient(create_app(notebook(guides=guides))) as c:
        source = add(c)
        assert source["guide_status"] == "failed" and "busy" in str(source["guide_error"])
        assert [x["kind"] for x in source["case_studies"]] == ["off_topic"]  # type: ignore[attr-defined]
        assert ask(c, "zebrafish")["status"] == "answered"
    assert "zebrafish" in guides.texts[0]


def test_answers_come_only_from_the_selected_source(client: TestClient) -> None:
    add(client)
    add(client, ("other.md", (FILLER + "Zebrafish bananas are yellow.").encode(), "text/markdown"))
    data = ask(client, "zebrafish", strategy="sparse+rerank")
    assert data["status"] == "answered"
    assert {p["source_filename"] for p in data["passages"]} == {"notes.txt"}  # type: ignore[attr-defined]
    assert data["sources"][0]["title"] == "notes.txt"  # type: ignore[index]


def test_gate_refusal_retrieval_only_and_outage() -> None:
    with TestClient(create_app(notebook(score=-9.0))) as c:
        add(c)
        data = ask(c, OFF_TOPIC_QUESTION)
        assert data["status"] == "refused_by_gate" and data["gate"]["passed"] is False  # type: ignore[index]
        assert ask(c, "x", generate=False)["status"] == "refused_by_gate"
    with TestClient(create_app(notebook())) as c:
        add(c)
        data = ask(c, "zebrafish", strategy="sparse", generate=False)
        assert data["status"] == "retrieval_only" and data["gate"] is None
    outage = StubClient(error=GenerationUnavailableError("busy", status_code=503))
    with TestClient(create_app(notebook(client=outage))) as c:
        add(c)
        data = ask(c, "zebrafish")
        assert data["status"] == "error" and "503" in str(data["error"]) and data["passages"]


def test_removing_a_source(client: TestClient) -> None:
    add(client)
    assert client.delete("/api/sources/notes.txt").status_code == 204
    assert client.get("/api/sources").json() == []
    assert client.delete("/api/sources/notes.txt").status_code == 404


def test_bad_uploads_and_queries_are_rejected(client: TestClient) -> None:
    assert client.post("/api/sources", files={"file": ("x.exe", b"MZ", "a/b")}).status_code == 415
    add(client)
    dup = client.post("/api/sources", files={"file": NOTES})
    assert dup.status_code == 422 and "already" in dup.json()["detail"]
    broken = client.post("/api/sources", files={"file": ("b.pdf", b"not a pdf", "application/pdf")})
    assert broken.status_code == 422
    bad = {"question": "x", "strategy": "nope", "sources": ["notes.txt"]}
    assert client.post("/api/query", json=bad).status_code == 422
    blank = {"question": "   ", "sources": ["notes.txt"]}
    assert client.post("/api/query", json=blank).status_code == 422


def test_a_client_supplied_path_is_reduced_to_its_filename(client: TestClient) -> None:
    source = add(client, ("../../evil.txt", b"zebrafish", "text/plain"))
    assert source["filename"] == "evil.txt"
