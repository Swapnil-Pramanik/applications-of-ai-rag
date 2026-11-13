"""The HTTP layer over DemoService, with the same stubs as test_ui_service."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app.generation.gemini import GenerationUnavailableError
from tests.unit.test_ui_service import StubClient, service, uploading_service


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(service=uploading_service())) as c:
        yield c


def test_health_and_meta(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "chunks": 3}
    meta = client.get("/api/meta").json()
    assert meta["chunk_count"] == 3
    assert {s["key"] for s in meta["strategies"]} >= {"hybrid+rerank", "sparse"}
    assert meta["examples"] and all(p["uploaded"] is False for p in meta["papers"])


def test_an_answered_query_has_numbered_claims_sources_and_gate_marks(client: TestClient) -> None:
    body = client.post("/api/query", json={"question": "adam beta2", "strategy": "sparse+rerank"})
    assert body.status_code == 200
    data = body.json()
    assert data["status"] == "answered"
    assert data["claims"] and data["claims"][0]["sources"] == [1]
    assert data["sources"][0]["number"] == 1 and data["sources"][0]["title"]
    assert data["gate"]["passed"] is True
    assert all(p["passed_gate"] is True for p in data["passages"])
    assert [p["rank"] for p in data["passages"]] == list(range(1, len(data["passages"]) + 1))


def test_the_gate_refusal_comes_through_with_a_reason() -> None:
    with TestClient(create_app(service=service(score=-9.0))) as c:
        data = c.post("/api/query", json={"question": "capital of france"}).json()
    assert data["status"] == "refused_by_gate"
    assert data["claims"] == [] and data["gate"]["passed"] is False and data["gate"]["reason"]


def test_retrieval_only_and_uncalibrated_strategies(client: TestClient) -> None:
    data = client.post(
        "/api/query", json={"question": "adam", "strategy": "sparse", "generate": False}
    ).json()
    assert data["status"] == "retrieval_only"
    assert data["gate"] is None and all(p["passed_gate"] is None for p in data["passages"])


def test_an_outage_is_reported_as_an_error_with_retrieval() -> None:
    outage = StubClient(error=GenerationUnavailableError("busy", status_code=503))
    with TestClient(create_app(service=service(client=outage))) as c:
        data = c.post("/api/query", json={"question": "adam"}).json()
    assert data["status"] == "error" and "503" in data["error"] and data["passages"]


def test_bad_queries_are_rejected(client: TestClient) -> None:
    assert client.post("/api/query", json={"question": "x", "strategy": "nope"}).status_code == 422
    assert client.post("/api/query", json={"question": "   "}).status_code == 422
    assert client.post("/api/query", json={"question": ""}).status_code == 422


def test_upload_makes_a_document_searchable(client: TestClient) -> None:
    files = {"file": ("../../notes.txt", b"the zebrafish optimizer is great", "text/plain")}
    up = client.post("/api/upload", files=files)
    assert up.status_code == 200
    assert up.json()["filename"] == "notes.txt" and up.json()["chunk_count"] > 3

    meta = client.get("/api/meta").json()
    assert {"title": "notes.txt", "chunks": up.json()["chunks"], "uploaded": True} in meta["papers"]
    data = client.post(
        "/api/query", json={"question": "zebrafish", "strategy": "sparse", "generate": False}
    ).json()
    assert data["passages"][0]["source_filename"] == "notes.txt"


def test_bad_uploads_are_rejected(client: TestClient) -> None:
    exe = {"file": ("evil.exe", b"MZ", "application/octet-stream")}
    assert client.post("/api/upload", files=exe).status_code == 415
    dup = {"file": ("1412.6980.pdf", b"%PDF", "application/pdf")}
    assert "already" in client.post("/api/upload", files=dup).json()["detail"]
    broken = {"file": ("broken.pdf", b"not a pdf", "application/pdf")}
    assert client.post("/api/upload", files=broken).status_code == 422
