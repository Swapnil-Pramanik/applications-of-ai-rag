.PHONY: setup demo api web test lint format typecheck corpus validate-golden eval eval-ablation eval-sweep eval-calibrate

setup:
	uv sync

# Streamlit demo on http://localhost:8501. `python -m` puts the repo root on
# the import path so `app`, `eval` and `ui` import normally.
demo:
	uv run python -m streamlit run ui/streamlit_app.py

# JSON API for the Next.js frontend on http://localhost:8000 (interactive docs at /docs).
api:
	uv run uvicorn api.main:app --port 8000

# Next.js presentation UI on http://localhost:3000 (needs `make api` running).
web:
	cd web && npm install && npm run dev

test:
	uv run pytest --cov=app --cov=eval --cov=ui --cov=api --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app eval scripts ui api

# Downloads the 18 arXiv PDFs into data/papers/ (skip if already present).
corpus:
	uv run python scripts/fetch_corpus.py

validate-golden:
	uv run python -m eval.harness.validate_golden

# Headline retrieval number (hybrid + cross-encoder rerank).
eval:
	uv run python -m eval.harness.runner --strategy hybrid+rerank --require-verified 1.0

# Six-arm retrieval ablation: each +rerank arm wraps the same base retriever as
# the row above it, so the delta is attributable to the reranker alone.
eval-ablation:
	uv run python -m eval.harness.runner --strategy dense --require-verified 1.0
	uv run python -m eval.harness.runner --strategy sparse --require-verified 1.0
	uv run python -m eval.harness.runner --strategy hybrid --require-verified 1.0
	uv run python -m eval.harness.runner --strategy dense+rerank --require-verified 1.0
	uv run python -m eval.harness.runner --strategy sparse+rerank --require-verified 1.0
	uv run python -m eval.harness.runner --strategy hybrid+rerank --require-verified 1.0

# Chunk size then overlap: six arms, strategy held at hybrid+rerank. Writes
# eval/results/chunk-sweep.md plus one JSON per arm (losers included).
eval-sweep:
	uv run python -m eval.harness.sweep

# Sweeps the abstention threshold. Retrieval-side only: zero Gemini calls.
eval-calibrate:
	uv run python -m eval.harness.calibrate
