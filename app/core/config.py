from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"

    # Embeddings. bge-small-en-v1.5 is 384-dim; changing the model almost
    # certainly changes the dimension, and a dimension change invalidates
    # every persisted index — so both are config, and the index build
    # asserts they agree.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # Cross-encoder reranker. Scores (query, passage) jointly, so it only ever
    # sees `candidate_k` pairs at query time — see retrieval/reranker.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # FAISS / BM25 index files live here.
    index_dir: Path = Path("data/indexes")
    index_cache_size: int = 4

    # Chunking. Selected by the Day 9 sweep, not guessed: 600/90 won on
    # nDCG@10 (0.539 vs 0.487 for the previous 1000/150) at equal Recall@5,
    # and 15% overlap beat both 0% and 30% at that size. See
    # eval/results/chunk-sweep.md for all six arms, losers included.
    chunk_size: int = 600
    chunk_overlap: int = 90

    # Retrieval. `candidate_k` is the per-retriever pool handed to fusion;
    # `top_k` is what survives to the generator. `rrf_k` damps how much a
    # rank-1 hit outweighs agreement between retrievers — see retrieval/fusion.
    # All three are swept by the Day 8 ablation, so none of them is a constant.
    candidate_k: int = 20
    top_k: int = 5
    rrf_k: int = 60

    # Generation. Pinned to an explicit version, never `gemini-flash-latest`:
    # a floating alias would silently change the model underneath committed
    # RAGAS numbers and make the eval non-reproducible.
    #
    # `thinking_level` is not a default we inherited — 3.6-flash does
    # chain-of-thought by default, and measured on a realistic 5-chunk RAG
    # prompt that costs 4.14s mean vs 2.49s at "LOW". Default thinking blows
    # the 3.0s end-to-end reference point on its own, before retrieval.
    # (`thinking_budget: 0` is rejected with HTTP 400 on this model.)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_thinking_level: str = "LOW"
    # Zero temperature so the same question over the same chunks gives the same
    # answer — the eval harness cannot measure faithfulness against a moving target.
    gemini_temperature: float = 0.0
    gemini_max_output_tokens: int = 1024

    # Which hosted LLM answers questions and writes source guides. "groq" runs
    # the open-weight gpt-oss-120b on Groq's free tier (~1,000 requests/day);
    # "gemini" is the original provider (free tier: 20 requests/day). Retrieval,
    # the gate and every benchmark number are identical under both.
    llm_provider: str = "groq"
    # Accepts GROQ_API_KEY (conventional) or GROQ_KEY.
    groq_api_key: str = Field(default="", validation_alias=AliasChoices("GROQ_API_KEY", "GROQ_KEY"))
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 30.0
    # Groq's free tier caps gpt-oss-120b at 8,000 tokens per minute, counting the
    # prompt plus the reserved reply budget. A guide therefore reads only the
    # first ~16k characters (~4-5k tokens), and replies are capped so one
    # question costs ~2k tokens (3-4 questions a minute).
    groq_guide_max_chars: int = 16_000
    groq_answer_max_tokens: int = 1024
    groq_guide_max_tokens: int = 1200

    # Upstream failure handling. The SDK retries 5xx with backoff by default,
    # and left unbounded that turned a single 503 into a 128s hang in a Day 10
    # measurement — worse for a user than failing fast. Two attempts inside a
    # hard per-request timeout keeps a bad upstream from holding a worker.
    gemini_timeout_seconds: float = 20.0
    gemini_retry_attempts: int = 2

    # Abstention gate, retrieval half (Q6). Thresholds apply to cross-encoder
    # logits, which are unbounded and roughly comparable across queries —
    # they are meaningless against cosine, BM25 or RRF scores.
    #
    # Selected by `make eval-calibrate` from a 12-point sweep — see
    # eval/results/abstention-calibration.md. At -2.0 the gate admits 95% of
    # answerable golden questions while refusing 92.9% of the adversarial
    # questions it is meant to refuse.
    #
    # Why -2.0 is defensible even though the adversarial set is not yet
    # human-verified: gate-catch is **flat at 0.929 across the whole -2.0..+2.0
    # plateau**, while answerable-pass falls 0.950 -> 0.783 over the same span.
    # So the choice *within* the plateau is decided entirely by the 60/60
    # human-verified golden set, and does not depend on the unverified labels.
    # The catch rate itself is not quotable until verification lands.
    confidence_threshold: float = -2.0
    min_supporting_chunks: int = 1

    # Abstention policy (the Q6 hallucination gate, model-side half).
    # When true, a claim whose citations all fail to resolve invalidates the
    # whole answer and the service abstains. Strict on purpose: an uncited
    # claim is exactly the failure mode this project promises not to ship.
    require_citation_per_claim: bool = True

    @field_validator("llm_provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        if v not in ("groq", "gemini"):
            raise ValueError(f"llm_provider must be 'groq' or 'gemini', got {v!r}")
        return v

    @field_validator("gemini_thinking_level")
    @classmethod
    def _known_thinking_level(cls, v: str) -> str:
        """Reject a typo at startup rather than at request time.

        `types.ThinkingLevel("lo")` does not raise — it warns and hands back a
        synthetic `ThinkingLevel.lo` member that sails through to the API. That
        is the silent-misconfiguration failure mode this project keeps guarding
        against, so the allowed set is pinned here instead.
        """
        allowed = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
        level = v.strip().upper()
        if level not in allowed:
            raise ValueError(f"gemini_thinking_level must be one of {sorted(allowed)}, got {v!r}")
        return level


settings = Settings()
