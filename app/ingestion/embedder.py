"""Dense embeddings via a self-hosted sentence-transformers model.

Three details here are load-bearing and worth being able to defend:

1. **Asymmetric prefixing.** BGE is trained with an instruction prefix on the
   *query* side only. Passages are embedded bare. Prefixing both sides (or
   neither) still "works" in the sense that it returns results, but it pushes
   queries and passages into slightly different regions of the space and
   quietly costs recall — there is no error to notice.

2. **L2 normalization.** Vectors are unit-normalized at embed time so that
   FAISS `IndexFlatIP` (inner product) computes exact cosine similarity.
   Normalizing here rather than at search time means the index can never hold
   an un-normalized vector.

3. **Lazy singleton.** Loading the model costs ~1s and ~130MB. Importing this
   module must stay cheap so that unit tests, Alembic, and the FastAPI import
   graph don't pay for a model they never use.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from app.core.config import settings

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point
    from sentence_transformers import SentenceTransformer

Vector = npt.NDArray[np.float32]

# The prefix BAAI publishes for bge-*-en-v1.5 retrieval. Queries only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """Turns text into unit-normalized float32 vectors.

    `model` is injectable so tests can pass a stub and stay offline.
    """

    def __init__(self, model: SentenceTransformer | None = None, model_name: str | None = None):
        self._model = model
        self._model_name = model_name or settings.embedding_model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> Vector:
        vectors = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> Vector:
        """Embed passages for indexing. No instruction prefix — see module docstring."""
        if not texts:
            return np.empty((0, settings.embedding_dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> Vector:
        """Embed a single search query, with BGE's retrieval instruction prefix."""
        vector: Vector = self._encode([QUERY_INSTRUCTION + text])[0]
        return vector


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
