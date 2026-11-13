"""FastAPI app: `make api` serves it on http://localhost:8000 (docs at /docs).

Four routes, all thin wrappers over `DemoService`, so the API answers exactly
like the Streamlit demo and the eval harness. Deliberately no auth, database,
rate limiting or streaming: this is a local demo backend, not a public service.

The service (corpus + both models) loads once at startup and lives on
`app.state`; tests swap in a stub through `create_app(service=...)`.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ExampleOut,
    MetaResponse,
    Paper,
    QueryRequest,
    QueryResponse,
    StrategyOut,
    UploadResponse,
    to_response,
)
from app.ingestion.parsers import UnsupportedFormatError
from ui.service import EXAMPLES, STRATEGY_LABELS, DemoService, UploadError, load_service

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


def get_service(request: Request) -> DemoService:
    service: DemoService = request.app.state.service
    return service


Service = Annotated[DemoService, Depends(get_service)]


def create_app(service: DemoService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service if service is not None else load_service()
        yield

    app = FastAPI(title="Ask the Papers API", version="1.0", lifespan=lifespan)
    # The Next.js dev server proxies /api to here, so CORS is only needed when
    # the frontend calls the API directly from another port.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(svc: Service) -> dict[str, object]:
        return {"status": "ok", "chunks": svc.chunk_count}

    @app.get("/api/meta", response_model=MetaResponse)
    def meta(svc: Service) -> MetaResponse:
        uploaded = set(svc.uploaded)
        return MetaResponse(
            papers=[
                Paper(title=title, chunks=count, uploaded=title in uploaded)
                for title, count in svc.papers()
            ],
            chunk_count=svc.chunk_count,
            strategies=[StrategyOut(key=k, label=v) for k, v in STRATEGY_LABELS.items()],
            examples=[ExampleOut(label=e.label, question=e.question) for e in EXAMPLES],
        )

    @app.post("/api/query", response_model=QueryResponse)
    def query(body: QueryRequest, svc: Service) -> QueryResponse:
        if body.strategy not in STRATEGY_LABELS:
            raise HTTPException(422, f"unknown strategy {body.strategy!r}")
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "question is empty")
        return to_response(svc.ask(question, body.strategy, generate=body.generate))

    @app.post("/api/upload", response_model=UploadResponse)
    async def upload(file: UploadFile, svc: Service) -> UploadResponse:
        # Path(...).name drops any directory part a client sends in the filename.
        name = Path(file.filename or "").name
        if not name.lower().endswith(ALLOWED_SUFFIXES):
            raise HTTPException(415, f"supported file types: {', '.join(ALLOWED_SUFFIXES)}")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file is larger than 20 MB")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                # Parsing and embedding are CPU-bound; keep them off the event loop.
                result = await run_in_threadpool(svc.add_document, path)
            except (UploadError, UnsupportedFormatError) as error:
                raise HTTPException(422, str(error)) from error
            except Exception as error:  # a corrupt PDF raises from deep inside PyMuPDF
                raise HTTPException(422, f"could not read {name}: {error}") from error
        return UploadResponse(
            filename=result.filename, chunks=result.chunks, chunk_count=svc.chunk_count
        )

    return app


app = create_app()
