# Applications of AI Project

**Cited question answering over ML research papers.** Ask a natural-language question across 18
landmark ML papers (Transformer, BERT, ResNet, GAN, Adam, LoRA, RAG, PPO, …) and get an answer
whose every claim cites a source paper, page and passage — or an explicit refusal when the papers
don't support an answer.

Built as a semester project for *Applications of AI*. It is a deliberately scoped-down copy of a
larger RAG service: the multi-tenant API, database, caching, tracing and cloud deployment were cut
so a team of 1–2 can finish, understand and present it in one month.

## Measured results (60 human-verified questions, 600/90 chunking)

| Retriever | R@1 | **R@5** | R@10 | MRR@10 | nDCG@10 | Multi-hop R@5 |
|---|---|---|---|---|---|---|
| dense (BGE + FAISS) | 0.150 | 0.317 | 0.417 | 0.228 | 0.315 | 0.000 |
| sparse (BM25) | 0.250 | 0.483 | 0.600 | 0.351 | 0.455 | 0.083 |
| hybrid (RRF) | 0.167 | 0.450 | 0.583 | 0.290 | 0.408 | 0.083 |
| dense + rerank | 0.300 | 0.467 | 0.517 | 0.364 | 0.463 | 0.083 |
| sparse + rerank | 0.317 | 0.567 | 0.683 | 0.416 | 0.523 | 0.167 |
| **hybrid + rerank** (shipped) | 0.317 | **0.567** | 0.650 | 0.416 | **0.539** | 0.167 |

Reproduce with `make eval-ablation`; raw results live in `eval/results/`.

What the table says, honestly:

- **The cross-encoder reranker is the biggest single win** — it lifts every base retriever.
- **Plain RRF hybrid lost to BM25** (0.450 vs 0.483). RRF fuses *ranks*, so a mediocre dense
  rank-1 counts as much as a confident BM25 rank-1. Hybrid only earns its place after reranking,
  where it ties sparse+rerank on Recall@5 and leads on nDCG@10.
- **Multi-hop questions are the weak spot (0.167)**, and that is structural: they need two
  passages from two different papers in the same top 5, and every retriever here scores chunks
  independently. Fixing it needs query decomposition or diversity-aware retrieval — out of scope.
- Abstention threshold `-2.0` was selected by `make eval-calibrate`
  (`eval/results/abstention-calibration.md`). The catch rate is **not quotable yet**: the 40
  adversarial questions still await human verification.

## Features

| Feature | Status | Where |
|---|---|---|
| PDF / DOCX / TXT / MD parsing with page numbers | Done | `app/ingestion/parsers.py` |
| Recursive chunking (600 chars, 90 overlap) | Done | `app/ingestion/chunker.py` |
| Dense retrieval — `bge-small-en-v1.5` + FAISS | Done | `app/ingestion/embedder.py`, `app/retrieval/dense.py` |
| Sparse retrieval — BM25 | Done | `app/retrieval/sparse.py` |
| Hybrid fusion — Reciprocal Rank Fusion | Done | `app/retrieval/fusion.py` |
| Cross-encoder reranking | Done | `app/retrieval/reranker.py` |
| Abstention gate on reranker scores | Done | `app/retrieval/confidence.py` |
| Gemini answers with per-claim citations | Done | `app/generation/` |
| Evaluation harness (Recall@k, MRR, nDCG, per-type) | Done | `eval/harness/` |
| Streamlit demo: examples, retriever switch, cited answer, top-5 passages with gate marks | Done | `ui/` |
| Polished presentation UI | Planned (if time) | — |
| Verified abstention rate (40 adversarial Qs) | Planned | `eval/golden_set/adversarial.jsonl` |

## Architecture

```
            ┌──────────────── ingestion (app/ingestion) ────────────────┐
 PDFs ──▶ parse ──▶ chunk ──▶ embed (BGE) ──▶ FAISS index + BM25 index
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
                       app/generation/citations.py maps numbers ─▶ paper + page + quote

 eval/harness:  golden questions ─▶ same retrieval code ─▶ Recall@k / MRR / nDCG ─▶ eval/results/
 ui/:           streamlit_app.py (draws) ─▶ service.py (decides) ─▶ QueryPipeline
```

Layers talk only through small interfaces: every retriever implements
`Retriever.retrieve(query, k) -> list[RetrievedChunk]` (`app/retrieval/base.py`), and chunk text is
looked up through the `ChunkStore` protocol. The eval harness builds its retrievers from the same
ingestion code the app uses, so the measured numbers describe the real pipeline.

## Project structure

```
.
├── app/
│   ├── core/
│   │   ├── config.py          # all settings (env / .env), incl. chunking + threshold
│   │   └── corpus.py          # the 18 arXiv IDs -> titles
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
│   │   └── citations.py       # resolve cited numbers back to real chunks
│   └── pipeline.py            # QueryPipeline: retrieve -> gate -> generate
├── eval/
│   ├── golden_set/
│   │   ├── questions.jsonl    # 60 verified Q/A with verbatim supporting spans
│   │   ├── adversarial.jsonl  # 40 unanswerable questions (verification pending)
│   │   └── README.md          # labelling + verification protocol
│   ├── harness/
│   │   ├── golden_set.py      # loader + text normalisation
│   │   ├── adversarial.py     # adversarial loader
│   │   ├── metrics.py         # Recall@k, MRR, nDCG
│   │   ├── corpus_index.py    # ingests the corpus (cached) and builds each retriever
│   │   ├── runner.py          # `python -m eval.harness.runner --strategy ...`
│   │   ├── calibrate.py       # abstention-threshold sweep (no Gemini calls)
│   │   └── validate_golden.py # checks every span really occurs on its page
│   └── results/               # committed JSON / markdown results
├── ui/
│   ├── service.py             # demo logic, no Streamlit: DemoService, examples, labels
│   └── streamlit_app.py       # the Streamlit page (drawing only; not `app.py`, see below)
├── scripts/fetch_corpus.py    # downloads the 18 PDFs into data/papers/
├── tests/
│   ├── unit/                  # fast, no model downloads
│   ├── integration/           # real models / corpus; skipped when unavailable
│   └── fixtures/
├── data/papers/               # PDFs (gitignored, re-downloadable)
├── .github/workflows/ci.yml   # ruff, format, mypy, golden-set schema, pytest + coverage
├── Makefile
├── pyproject.toml / uv.lock
└── .env.example
```

## Layer breakdown

| Layer | Responsibility | Key types / functions |
|---|---|---|
| `app/core` | Configuration and corpus list | `Settings`, `settings`, `CORPUS` |
| `app/ingestion` | Turn files into embedded chunks and indexes | `parse_document`, `chunk_pages`, `Embedder`, `build_index`, `ingest_document` |
| `app/retrieval` | Rank chunks for a question; decide whether the evidence is good enough | `DenseRetriever`, `SparseRetriever`, `HybridRetriever`, `RerankingRetriever`, `ConfidenceGate` |
| `app/generation` | Ask Gemini for a structured, cited answer and verify its citations | `generate_answer`, `GeminiClient`, `assemble`, `Answer` |
| `app/pipeline.py` | One question end to end | `QueryPipeline.answer() -> AnswerResult` |
| `eval/harness` | Measure retrieval and calibrate the gate | `load_corpus`, `EvalCorpus.build_retriever`, `run`, `sweep` |
| `ui/service.py` | Load corpus + models once, run a question through a chosen retriever, report the outcome | `DemoService.ask() -> DemoResult`, `EXAMPLES`, `numbered_claims` |
| `ui/streamlit_app.py` | Render settings, examples, answer, sources, retrieved passages | Streamlit only, no logic |

## Request flow

1. `load_corpus()` (`eval/harness/corpus_index.py`) ingests `data/papers/*.pdf`, caching chunks +
   vectors in `eval/.cache/` keyed by chunking params, model and file fingerprints.
2. `EvalCorpus.build_retriever("hybrid+rerank")` builds FAISS + BM25 → RRF → cross-encoder.
3. `QueryPipeline(retriever, gate=ConfidenceGate()).answer(question)`:
   - retrieves the top 5 reranked chunks;
   - **gate**: if the best cross-encoder score is below `confidence_threshold` (−2.0) → abstain
     with `abstain_source="gate"`, no Gemini call;
   - otherwise Gemini returns JSON claims citing source *numbers*; `citations.py` maps them to
     real chunks, drops invented ones, and abstains (`abstain_source="model"`) if any claim is
     left uncited.
4. A Gemini outage (429/503) raises `GenerationUnavailableError` — never counted as an abstention.

**Demo flow** (`make demo`): `get_service()` (cached with `st.cache_resource`) calls
`load_service()` once. Each question goes through `DemoService.ask(question, strategy, generate)`,
which returns a `DemoResult` with one of five statuses: `answered`, `refused_by_gate`,
`refused_by_model`, `retrieval_only` (generation toggled off, uses no quota) or `error` (Gemini
outage; retrieval is still shown). The confidence gate only runs for `+rerank` retrievers, because
its threshold is a cross-encoder score and means nothing against cosine/BM25/RRF scores.

The page is `ui/streamlit_app.py`, not `ui/app.py`: Streamlit adds `ui/` to the import path, and a
file named `app.py` there would shadow the `app` package.

## Key data models

- `RetrievedChunk` — `vector_id, text, score, page_number, chunk_index, source_filename`.
- `GoldenQuestion` — id, question, type (`factual | numeric | comparative | multi_hop`),
  relevance (`any | all`), and spans of `(source pdf, page, verbatim quote)`. Relevance is judged
  by span text, not chunk id, so one label set grades every chunking fairly.
- `Answer` → `claims[]` → `Citation(source_filename, page_number, chunk_index, quote)`;
  an unanswered `Answer` carries `abstain_reason`.
- `AnswerResult` — `answer`, `gate: GateDecision`, `retrieved` (all top-k chunks, including ones
  the gate dropped), `abstain_source` (`gate` / `model` / `None`).
- `DemoResult` (`ui/service.py`) — `question, strategy, retrieved, gate | None, answer | None,
  error | None, seconds`, plus `status` and `passed_gate(chunk)`.

## External services and data

| Service | Used for | Cost |
|---|---|---|
| Google Gemini (`gemini-3.6-flash`, thinking LOW) | answer generation | free tier — 429/503s happen; don't bulk-test |
| Hugging Face Hub | one-time download of `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` | free |
| arXiv | one-time PDF download (`make corpus`) | free |

No database. Everything runs in memory on a laptop; both models are small and CPU-only.

## Dependencies

Runtime: `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `pymupdf`, `python-docx`,
`google-genai`, `pydantic-settings`, `structlog`, `streamlit`. Dev: `pytest`, `pytest-asyncio`, `pytest-cov`,
`ruff`, `mypy`. Managed with `uv` (Python 3.11).

## Configuration

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. That is the only secret; `.env` is
gitignored. Every other setting (chunk size/overlap, `top_k`, `candidate_k`, `rrf_k`,
`confidence_threshold`, model names) has a default in `app/core/config.py` and can be overridden
by an environment variable of the same name in upper case.

## Running it

```bash
uv sync                 # install
make corpus             # download the 18 PDFs (once)
make demo               # Streamlit demo at http://localhost:8501 (first load ~20-30s)
make test               # unit + integration tests, 70% coverage gate
make eval               # headline Recall@k for hybrid+rerank
make eval-ablation      # all six retrievers
make eval-calibrate     # abstention threshold sweep (no API calls)
```
