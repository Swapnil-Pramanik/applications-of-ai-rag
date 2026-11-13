.PHONY: setup demo test lint format typecheck corpus validate-golden eval eval-ablation eval-calibrate

setup:
	uv sync

# Streamlit demo on http://localhost:8501. `python -m` puts the repo root on
# the import path so `app`, `eval` and `ui` import normally.
demo:
	uv run python -m streamlit run ui/streamlit_app.py

test:
	uv run pytest --cov=app --cov=eval --cov=ui --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app eval scripts ui

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

# Sweeps the abstention threshold. Retrieval-side only: zero Gemini calls.
eval-calibrate:
	uv run python -m eval.harness.calibrate
