# Grounded Notebook — Applications of AI Project

**A NotebookLM-style notebook that answers only from your own documents.** Upload PDFs, Word files
or notes, tick which sources to use, and ask questions. Every claim cites a source, page and
passage, and when your sources don't contain the answer the notebook **refuses instead of
guessing**. For each upload, Gemini writes a short guide and five case-study questions (fact,
comparison, multi-hop, off-topic, not-in-document) that show each behaviour live on *that* file.

Built as a semester project for *Applications of AI*. Under the hood it is a retrieval-augmented
generation (RAG) pipeline: hybrid retrieval (BGE + FAISS dense, BM25 sparse, Reciprocal Rank
Fusion), a cross-encoder reranker, a confidence gate, and Gemini for cited answers.

**Nothing is trained or fine-tuned.** All three models (BGE embeddings, the MiniLM cross-encoder,
Gemini) are used as-is. The 18 ML papers in `data/papers/` are not training data: they are the
**benchmark** the pipeline is measured on (60 hand-verified questions, below). The notebook app
never shows them; it only answers from what you upload.

It is a scoped-down copy of a larger RAG service: multi-tenant auth, database, caching, tracing
and cloud deployment were cut so a team of 1–2 can finish, understand and present it in a month.

## How we know it works: measured on 18 ML papers (60 human-verified questions, 600/90 chunking)

| Retriever | R@1 | **R@5** | R@10 | MRR@10 | nDCG@10 | Multi-hop R@5 |
|---|---|---|---|---|---|---|
| dense (BGE + FAISS) | 0.150 | 0.317 | 0.417 | 0.228 | 0.315 | 0.000 |
| sparse (BM25) | 0.250 | 0.483 | 0.600 | 0.351 | 0.455 | 0.083 |
| hybrid (RRF) | 0.167 | 0.450 | 0.583 | 0.290 | 0.408 | 0.083 |
| dense + rerank | 0.300 | 0.467 | 0.517 | 0.364 | 0.463 | 0.083 |
| sparse + rerank | 0.317 | 0.567 | 0.683 | 0.416 | 0.523 | 0.167 |
| **hybrid + rerank** (shipped) | 0.317 | **0.567** | 0.650 | 0.416 | **0.539** | 0.167 |

Reproduce with `make eval-ablation`; raw results live in `eval/results/`. The notebook runs this
exact pipeline (same code, same settings) on your uploads, so the table describes what it does.

What the table says, honestly:

- **The cross-encoder reranker is the biggest single win** — it lifts every base retriever.
- **Plain RRF hybrid lost to BM25** (0.450 vs 0.483). RRF fuses *ranks*, so a mediocre dense
  rank-1 counts as much as a confident BM25 rank-1. Hybrid only earns its place after reranking,
  where it ties sparse+rerank on Recall@5 and leads on nDCG@10.
- **Multi-hop questions are the weak spot (0.167)**, and that is structural: they need two
  passages from different places in the same top 5, and every retriever here scores chunks
  independently. Fixing it needs query decomposition or diversity-aware retrieval — out of scope.
  The notebook's *Multi-hop* case study shows this live.
- **Chunking was chosen from data, not by default.** `make eval-sweep` tried 400/600/1000/1500
  characters at 15% overlap, then 0%/15%/30% overlap at the winning size
  (`eval/results/chunk-sweep.md`). 600/90 tied 1000/150 on Recall@5 (0.567) and won on nDCG@10
  (0.539 vs 0.487); dropping overlap to 0 cost 15 points of Recall@5. Losing arms are committed too.
- Abstention threshold `-2.0` was selected by `make eval-calibrate`
  (`eval/results/abstention-calibration.md`). The catch rate is **not quotable yet**: the 40
  adversarial questions still await human verification.

## Features

| Feature | Status | Where |
|---|---|---|
| Notebook UI: sources panel (upload, tick, remove), chat, notebook guide, settings | Done | `web/` |
| Answers only from the selected sources (per-selection FAISS + BM25 index) | Done | `ui/service.py` `DemoService.retriever(strategy, sources)` |
| Per-source guide: summary + 4 LLM-written case-study questions (+ fixed off-topic one) | Done | `app/generation/guide.py`, `api/notebook.py` |
| Five-step pipeline strip per answer (Retrieve → Fuse → Rerank → Gate → Answer) | Done | `web/src/components/Pipeline.tsx` |
| Clickable citations, evidence list with score-vs-gate bars | Done | `web/src/components/ChatTurn.tsx`, `Evidence.tsx` |
| "Answer with Gemini" off switch: retrieval only, no API quota | Done | `web/`, `DemoService.ask(generate=False)` |
| PDF / DOCX / TXT / MD parsing with page numbers | Done | `app/ingestion/parsers.py` |
| Recursive chunking (600 chars, 90 overlap) | Done | `app/ingestion/chunker.py` |
| Dense retrieval — `bge-small-en-v1.5` + FAISS | Done | `app/ingestion/embedder.py`, `app/retrieval/dense.py` |
| Sparse retrieval — BM25 | Done | `app/retrieval/sparse.py` |
| Hybrid fusion — Reciprocal Rank Fusion | Done | `app/retrieval/fusion.py` |
| Cross-encoder reranking | Done | `app/retrieval/reranker.py` |
| Abstention gate on reranker scores | Done | `app/retrieval/confidence.py` |
| Gemini answers with per-claim citations | Done | `app/generation/` |
| Evaluation harness (Recall@k, MRR, nDCG, per-type) on the 18-paper benchmark | Done | `eval/harness/` |
| Chunk size / overlap sweep (six arms, losers recorded) | Done | `eval/harness/sweep.py` |
| Streamlit demo over the 18-paper benchmark corpus | Done | `ui/streamlit_app.py` |
| Verified abstention rate (40 adversarial Qs) | Planned | `eval/golden_set/adversarial.jsonl` |

## Architecture

```
            ┌──────────────── ingestion (app/ingestion) ────────────────┐
 file ──▶ parse ──▶ chunk ──▶ embed (BGE) ──▶ FAISS index + BM25 index (per selected sources)
                                                   │
            ┌───────────── query (app/retrieval, app/pipeline.py) ─────┘
 question ──▶ dense top-20 ─┐
             sparse top-20 ─┴─▶ RRF fusion ─▶ cross-encoder rerank (top 5)
                                                   │
                                          confidence gate ──▶ abstain (no LLM call)
                                                   │
                                                   ▼
                              Gemini (structured JSON, cites source numbers)
                                                   │
                       app/generation/citations.py maps numbers ─▶ file + page + quote

 web/ (Next.js, browser) ──HTTP JSON──▶ api/main.py ─▶ api/notebook.py NotebookService
                                                         ├─ ui/service.py DemoService (the pipeline above)
                                                         └─ app/generation/guide.py (1 Gemini call per upload)
 ui/streamlit_app.py ─▶ ui/service.py DemoService over the 18 benchmark papers
 eval/harness:  golden questions ─▶ same retrieval code ─▶ Recall@k / MRR / nDCG ─▶ eval/results/
```

One brain, three callers: the notebook API, the Streamlit demo and the eval harness all go through
the same retrievers and `QueryPipeline`, so the notebook cannot answer differently from what the
eval measures.

Layers talk only through small interfaces: every retriever implements
`Retriever.retrieve(query, k) -> list[RetrievedChunk]` (`app/retrieval/base.py`), and chunk text is
looked up through the `ChunkStore` protocol.

## Project structure

```
.
├── app/
│   ├── core/
│   │   ├── config.py          # all settings (env / .env), incl. chunking + threshold
│   │   └── corpus.py          # the 18 benchmark arXiv IDs -> titles
│   ├── ingestion/
│   │   ├── parsers.py         # PDF/DOCX/TXT/MD -> pages
│   │   ├── chunker.py         # recursive character splitter with overlap
│   │   ├── embedder.py        # sentence-transformers BGE wrapper (normalised vectors)
│   │   ├── index_builder.py   # FAISS IndexFlatIP build / save / load / search
│   │   └── pipeline.py        # parse -> chunk -> embed for one document
│   ├── retrieval/
│   │   ├── base.py            # RetrievedChunk + Retriever protocol
│   │   ├── chunk_store.py     # vector id -> chunk text/metadata
│   │   ├── dense.py           # FAISS retriever
│   │   ├── sparse.py          # BM25 retriever
│   │   ├── fusion.py          # Reciprocal Rank Fusion (HybridRetriever)
│   │   ├── reranker.py        # cross-encoder reranker
│   │   └── confidence.py      # abstention gate
│   ├── generation/
│   │   ├── prompts.py         # system instruction + numbered sources
│   │   ├── schema.py          # Pydantic model output + Answer/Claim/Citation
│   │   ├── gemini.py          # Gemini client, timeouts, outage error
│   │   ├── guide.py           # per-upload summary + case-study questions (one Gemini call)
│   │   └── citations.py       # resolve cited numbers back to real chunks
│   └── pipeline.py            # QueryPipeline: retrieve -> gate -> generate
├── api/
│   ├── main.py                # FastAPI app: health, meta, sources (list/add/remove), query
│   ├── notebook.py            # NotebookService: sources + background guides over DemoService
│   └── schemas.py             # Pydantic request/response models + DemoResult -> JSON
├── ui/
│   ├── service.py             # DemoService: corpus, uploads, source-scoped retrievers, ask()
│   └── streamlit_app.py       # Streamlit page over the 18 papers (not `app.py`, see below)
├── web/                       # Next.js 16 + TypeScript + Tailwind v4 notebook UI
│   ├── src/app/               # layout.tsx (fonts), globals.css (colour tokens), page.tsx
│   ├── src/lib/api.ts         # typed fetch client; types mirror api/schemas.py
│   ├── src/components/        # Notebook, SourcesPanel, ChatTurn, Pipeline, Evidence, StudioPanel
│   └── README.md              # per-file table for the frontend
├── eval/
│   ├── golden_set/
│   │   ├── questions.jsonl    # 60 verified Q/A with verbatim supporting spans
│   │   ├── adversarial.jsonl  # 40 unanswerable questions (verification pending)
│   │   └── README.md          # labelling + verification protocol
│   ├── harness/
│   │   ├── golden_set.py      # loader + text normalisation
│   │   ├── adversarial.py     # adversarial loader
│   │   ├── metrics.py         # Recall@k, MRR, nDCG
│   │   ├── corpus_index.py    # EvalCorpus: ingest (cached), subset/empty, build each retriever
│   │   ├── runner.py          # `python -m eval.harness.runner --strategy ...`
│   │   ├── sweep.py           # chunk size / overlap sweep -> results/chunk-sweep.md
│   │   ├── calibrate.py       # abstention-threshold sweep (no Gemini calls)
│   │   └── validate_golden.py # checks every span really occurs on its page
│   └── results/               # committed JSON / markdown results
├── scripts/fetch_corpus.py    # downloads the 18 benchmark PDFs into data/papers/
├── tests/
│   ├── unit/                  # fast, no model downloads, no network
│   ├── integration/           # real models / corpus; skipped when unavailable
│   └── fixtures/
├── data/papers/               # benchmark PDFs (gitignored, re-downloadable)
├── .github/workflows/ci.yml   # python: ruff, mypy, golden schema, pytest+cov; web: lint, build
├── Makefile
├── pyproject.toml / uv.lock
└── .env.example
```

## Layer breakdown

| Layer | Responsibility | Key types / functions |
|---|---|---|
| `app/core` | Configuration and the benchmark corpus list | `Settings`, `settings`, `CORPUS` |
| `app/ingestion` | Turn files into embedded chunks and indexes | `parse_document`, `chunk_pages`, `Embedder`, `build_index`, `ingest_document` |
| `app/retrieval` | Rank chunks for a question; decide whether the evidence is good enough | `DenseRetriever`, `SparseRetriever`, `HybridRetriever`, `RerankingRetriever`, `ConfidenceGate` |
| `app/generation` | Cited answers from Gemini (and verify citations); per-upload guides | `generate_answer`, `GeminiClient`, `assemble`, `Answer`, `GeminiGuideClient`, `SourceGuide` |
| `app/pipeline.py` | One question end to end | `QueryPipeline.answer() -> AnswerResult` |
| `eval/harness` | Measure retrieval and calibrate the gate; `EvalCorpus` also holds the notebook's chunks | `load_corpus`, `EvalCorpus.empty/subset/build_retriever`, `run`, `sweep` |
| `ui/service.py` | Hold a corpus + models, add/remove documents, build retrievers per (strategy, selected sources), run a question | `DemoService.ask(..., sources) -> DemoResult`, `add_document`, `remove_document`, `NoSourcesError` |
| `ui/streamlit_app.py` | Streamlit page over the 18 papers | Streamlit only, no logic |
| `api/notebook.py` | Sources in upload order, guides written on a background thread, case studies per source | `NotebookService`, `SourceInfo.case_studies`, `OFF_TOPIC_QUESTION` |
| `api/main.py` | HTTP wrapper; validates strategy, file type (pdf/docx/txt/md) and size (20 MB) | `create_app(notebook=None)`, `app` |
| `api/schemas.py` | Turn results into JSON the frontend only has to draw | `QueryRequest`, `QueryResponse`, `SourceOut`, `to_response`, `source_out` |
| `web/` | Draw the notebook. No answer logic | `Notebook`, `stagesFor`, `ask`, `addSource` (see `web/README.md`) |

## Request flow

**Notebook** (`make api` + `make web`):

1. The API starts with an empty notebook: `NotebookService(DemoService(EvalCorpus.empty()))`.
   Models load lazily on the first upload (~10 s once).
2. **Add source** — `POST /api/sources` (multipart `file`): written to a temp dir,
   `DemoService.add_document` parses → chunks (600/90) → embeds it and appends it under a lock,
   dropping cached retrievers. The response returns at once with `guide_status: "pending"`.
3. **Guide** — on a background thread, `GeminiGuideClient.generate_guide` reads the first 60k
   characters and returns a summary plus fact / comparison / multi-hop / not-in-document questions.
   The browser polls `GET /api/sources` every 2 s until no guide is pending. If Gemini is down or
   out of quota, the guide is `failed` and the source still works (only the off-topic case shows).
4. **Ask** — `POST /api/query {question, strategy, generate, sources}`.
   `DemoService.retriever(strategy, sources)` builds (and caches) a FAISS + BM25 index over *only*
   the selected files, then `QueryPipeline`:
   - retrieves the top 5 reranked chunks;
   - **gate**: if the best cross-encoder score is below `confidence_threshold` (−2.0) → refuse
     (`refused_by_gate`), no Gemini call;
   - otherwise Gemini returns JSON claims citing source *numbers*; `citations.py` maps them to real
     chunks, drops invented ones, and refuses (`refused_by_model`) if a claim is left uncited.
   - A Gemini outage (429/503) is `error`, never counted as a refusal; retrieval is still shown.
5. The browser draws the turn: pipeline strip (`stagesFor` marks each step done / skipped /
   stopped), claims with citation chips (click → citation panel), and the evidence list.

Statuses: `answered`, `refused_by_gate`, `refused_by_model`, `retrieval_only` (Gemini switched off),
`error`. The gate only runs for `+rerank` retrievers: its threshold is a cross-encoder score and
means nothing against cosine/BM25/RRF scores.

**Benchmark** (`make eval`): `load_corpus()` ingests `data/papers/*.pdf` (cached in
`eval/.cache/`), builds the same retrievers, and scores them against the golden set.

**Streamlit** (`make demo`): the earlier single-page demo over the 18 papers. The page is
`ui/streamlit_app.py`, not `ui/app.py`: Streamlit adds `ui/` to the import path, and a file named
`app.py` there would shadow the `app` package.

## Key data models

- `RetrievedChunk` — `vector_id, text, score, page_number, chunk_index, source_filename`.
- `EvalCorpus` — `chunks: list[EmbeddedChunk]`, `vectors`, `chunk_size`, `chunk_overlap`;
  `empty()`, `subset(filenames)`, `build_retriever(strategy)`.
- `GoldenQuestion` — id, question, type (`factual | numeric | comparative | multi_hop`),
  relevance (`any | all`), and spans of `(source pdf, page, verbatim quote)`.
- `Answer` → `claims[]` → `Citation(source_filename, page_number, chunk_index, quote)`;
  an unanswered `Answer` carries `abstain_reason`.
- `AnswerResult` — `answer`, `gate: GateDecision`, `retrieved`, `abstain_source` (`gate` / `model` / `None`).
- `DemoResult` (`ui/service.py`) — `question, strategy, retrieved, gate | None, answer | None,
  error | None, seconds`, plus `status` and `passed_gate(chunk)`.
- `SourceGuide` (`app/generation/guide.py`) — `summary, fact_question, comparison_question,
  multi_hop_question, not_in_document_question`. LLM suggestions, **not** verified labels: never
  used for a metric.
- `SourceInfo` / `GuideState` / `CaseStudy` (`api/notebook.py`) — a source's chunk count, guide
  status (`pending | ready | failed`), and its five case studies with the expected behaviour.
- `QueryResponse` / `SourceOut` (`api/schemas.py`) — the JSON the frontend draws.

## External services and data

| Service | Used for | Cost |
|---|---|---|
| Google Gemini (`gemini-3.6-flash`, thinking LOW) | cited answers (1 call per question) and source guides (1 call per upload) | **free tier: 20 requests/day** for this model, plus 429/503s under load. Plan the demo around it |
| Hugging Face Hub | one-time download of `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` | free |
| arXiv | one-time download of the 18 benchmark PDFs (`make corpus`) | free |

No database. Everything runs in memory on a laptop; both local models are small and CPU-only.
Uploaded sources and their guides live in memory and are gone when the API restarts.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | liveness, source and chunk counts |
| `/api/meta` | GET | retrievers, gate threshold, max upload size |
| `/api/sources` | GET | sources in upload order with guide status, summary, case studies |
| `/api/sources` | POST | multipart `file` (.pdf/.docx/.txt/.md, ≤20 MB) → `SourceOut` (guide pending) |
| `/api/sources/{filename}` | DELETE | remove a source (204; 404 if unknown) |
| `/api/query` | POST | `{question, strategy="hybrid+rerank", generate=true, sources=[...]}` → `QueryResponse`; 422 if no selected source has text |

**Known limitation:** on a document of only 1–2 chunks, BM25's IDF is ≤ 0 for every term, so the
BM25-only retriever finds nothing. The default hybrid retriever still finds it through dense
search, and real documents have dozens of chunks. The BM25 variant was not changed because that
would shift the measured benchmark numbers.

## Dependencies

Runtime: `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `pymupdf`, `python-docx`,
`google-genai`, `pydantic-settings`, `structlog`, `streamlit`, `fastapi`, `uvicorn`,
`python-multipart`. Frontend (`web/package.json`): `next` 16, `react` 19, `tailwindcss` 4,
`typescript`, `eslint`. Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx`, `ruff`, `mypy`.
Managed with `uv` (Python 3.11) and `npm` (Node 22+).

## Configuration

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. That is the only secret; `.env` is
gitignored. Every other setting (chunk size/overlap, `top_k`, `candidate_k`, `rrf_k`,
`confidence_threshold`, model names) has a default in `app/core/config.py` and can be overridden
by an environment variable of the same name in upper case.

Frontend: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). It is public by design and must
never hold a secret; the Gemini key stays in the API's `.env`.

## Running it

```bash
uv sync                 # install Python deps
make api                # notebook API at http://localhost:8000 (docs at /docs)
make web                # notebook UI at http://localhost:3000 (second terminal)

make corpus             # download the 18 benchmark PDFs (once; needed for eval + Streamlit)
make demo               # Streamlit demo over the benchmark papers at http://localhost:8501
make test               # unit + integration tests, 70% coverage gate
make eval               # headline Recall@k for hybrid+rerank
make eval-ablation      # all six retrievers
make eval-sweep         # chunk size / overlap sweep (re-ingests the corpus 6x, slow)
make eval-calibrate     # abstention threshold sweep (no API calls)
```

**Demo tip:** the free tier allows ~20 Gemini calls a day (each upload's guide is one, each
answered question is one). Upload your demo files before presenting and keep `make api` running: a
restart clears them, and re-uploading spends a guide call each. Keep "Answer with Gemini" off while
rehearsing and switch it on only for the questions you show. Gate refusals (e.g. the off-topic case)
never use quota.
