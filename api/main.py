"""FastAPI app for the notebook UI: `make api` serves it on http://localhost:8000 (docs at /docs).

NotebookLM-style: the notebook starts empty, you add sources, and every
question is answered only from the sources you select. Each route is a thin
wrapper over `NotebookService`, which wraps the same `DemoService` the eval
harness measures. Deliberately no auth, database, rate limiting or streaming:
this is a local demo backend, and state lives in memory until restart.
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

from api.notebook import NotebookService
from api.schemas import (
    MetaResponse,
    QueryRequest,
    QueryResponse,
    SourceOut,
    StrategyOut,
    source_out,
    to_response,
)
from app.core.config import settings
from app.ingestion.parsers import UnsupportedFormatError
from eval.harness.corpus_index import EvalCorpus
from ui.service import STRATEGY_LABELS, DemoService, NoSourcesError, UploadError

MAX_UPLOAD_MB = 20
ALLOWED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


def empty_notebook() -> NotebookService:
    return NotebookService(DemoService(EvalCorpus.empty()))


def get_notebook(request: Request) -> NotebookService:
    notebook: NotebookService = request.app.state.notebook
    return notebook


Notebook = Annotated[NotebookService, Depends(get_notebook)]


def create_app(notebook: NotebookService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.notebook = notebook if notebook is not None else empty_notebook()
        yield

    app = FastAPI(title="Notebook API", version="2.0", lifespan=lifespan)
    # The browser calls the API directly from the Next.js dev server.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(nb: Notebook) -> dict[str, object]:
        return {"status": "ok", "sources": len(nb.sources()), "chunks": nb.chunk_count}

    @app.get("/api/meta", response_model=MetaResponse)
    def meta() -> MetaResponse:
        return MetaResponse(
            strategies=[StrategyOut(key=k, label=v) for k, v in STRATEGY_LABELS.items()],
            gate_threshold=settings.confidence_threshold,
            max_upload_mb=MAX_UPLOAD_MB,
        )

    @app.get("/api/sources", response_model=list[SourceOut])
    def list_sources(nb: Notebook) -> list[SourceOut]:
        return [source_out(s) for s in nb.sources()]

    @app.post("/api/sources", response_model=SourceOut)
    async def add_source(file: UploadFile, nb: Notebook) -> SourceOut:
        # Path(...).name drops any directory part a client sends in the filename.
        name = Path(file.filename or "").name
        if not name.lower().endswith(ALLOWED_SUFFIXES):
            raise HTTPException(415, f"supported file types: {', '.join(ALLOWED_SUFFIXES)}")
        data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                # Parsing and embedding are CPU-bound; keep them off the event loop.
                await run_in_threadpool(nb.add_source, path)
            except (UploadError, UnsupportedFormatError) as error:
                raise HTTPException(422, str(error)) from error
            except Exception as error:  # a corrupt PDF raises from deep inside PyMuPDF
                raise HTTPException(422, f"could not read {name}: {error}") from error
        return next(source_out(s) for s in nb.sources() if s.filename == name)

    @app.delete("/api/sources/{filename}", status_code=204)
    def remove_source(filename: str, nb: Notebook) -> None:
        try:
            nb.remove_source(filename)
        except KeyError as error:
            raise HTTPException(404, f"no source named {filename!r}") from error

    @app.post("/api/query", response_model=QueryResponse)
    def query(body: QueryRequest, nb: Notebook) -> QueryResponse:
        if body.strategy not in STRATEGY_LABELS:
            raise HTTPException(422, f"unknown strategy {body.strategy!r}")
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "question is empty")
        try:
            result = nb.ask(question, body.strategy, generate=body.generate, sources=body.sources)
        except NoSourcesError as error:
            raise HTTPException(422, str(error)) from error
        return to_response(result)

    return app


app = create_app()
