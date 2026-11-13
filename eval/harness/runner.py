"""Runs a retrieval configuration over the golden set and records the numbers.

Usage:  `make eval`  (or `python -m eval.harness.runner --strategy hybrid`)

**One retrieval pass, many cutoffs.** The runner retrieves once at
`max(cutoffs)` depth and derives Recall@1/3/5/10 from the single
`first_satisfied_rank` that pass produced. Retrieving separately per cutoff
would be four times the work for identical answers, since every retriever here
returns a prefix-stable ranking.

**Unverified questions are counted and reported, never silently dropped.**
`human-verified: 60/60` is the current state of the golden set (see
`eval/golden_set/README.md`), and a Recall@5 computed over unverified labels
is a measurement of the labels as much as of the retriever. The runner
therefore stamps `verified_fraction` into every result file and refuses to
call a run authoritative below `--require-verified`. That flag defaults to 0.0
so the harness is usable during development — but the README must not quote a
number whose result file says `"authoritative": false`.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings
from app.retrieval.reranker import Reranker
from eval.harness.corpus_index import (
    DEFAULT_CORPUS_DIR,
    STRATEGIES,
    EvalCorpus,
    load_corpus,
)
from eval.harness.golden_set import QUESTIONS_PATH, GoldenQuestion, load_golden_set, normalize
from eval.harness.metrics import (
    AggregateScores,
    QuestionScore,
    aggregate,
    first_satisfied_rank,
    ndcg_at_k,
    relevant_chunk_count,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CUTOFFS = [1, 3, 5, 10]


@dataclass(frozen=True)
class RunConfig:
    """Everything that can change a number, in one hashable record.

    Written verbatim into the result file. A result whose configuration you
    cannot reconstruct is not evidence, and the ablation tables in the README
    are built by diffing these records against each other.
    """

    strategy: str = "hybrid"
    chunk_size: int = field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = field(default_factory=lambda: settings.chunk_overlap)
    candidate_k: int = field(default_factory=lambda: settings.candidate_k)
    rrf_k: int = field(default_factory=lambda: settings.rrf_k)
    embedding_model: str = field(default_factory=lambda: settings.embedding_model)

    @property
    def label(self) -> str:
        # "+" is legal in a filename but awkward in shells and URLs.
        strategy = self.strategy.replace("+", "-")
        return f"{strategy}-cs{self.chunk_size}-co{self.chunk_overlap}"


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    scores: AggregateScores
    per_question: list[QuestionScore]
    n_chunks: int
    verified_fraction: float
    authoritative: bool
    by_type: dict[str, dict[str, float]]
    timestamp: str
    platform: str

    def as_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "scores": self.scores.as_dict(),
            "by_type": self.by_type,
            "n_chunks": self.n_chunks,
            "verified_fraction": round(self.verified_fraction, 4),
            "authoritative": self.authoritative,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "per_question": [asdict(s) for s in self.per_question],
        }


def _score_question(
    question: GoldenQuestion,
    texts: list[str],
    sources: list[str],
    normalized_corpus: list[str],
    k: int,
) -> QuestionScore:
    normalized_retrieved = [normalize(text) for text in texts]
    n_relevant = relevant_chunk_count(question, normalized_corpus)
    return QuestionScore(
        question_id=question.id,
        question_type=question.type,
        rank=first_satisfied_rank(question, texts),
        ndcg=ndcg_at_k(question, normalized_retrieved, k, n_relevant),
        n_relevant=n_relevant,
        retrieved_sources=tuple(dict.fromkeys(sources)),
    )


def _breakdown(scores: list[QuestionScore], k: int) -> dict[str, dict[str, float]]:
    """Per-question-type recall, because the average hides where hybrid earns its keep.

    The interview claim this exists to support is specific: BM25 should help
    most on `numeric` and `factual` questions, where the answer hinges on an
    exact figure or identifier that a 384-dim embedding blurs. A single
    aggregate number cannot show that; this breakdown can.
    """
    out: dict[str, dict[str, float]] = {}
    for question_type in sorted({s.question_type for s in scores}):
        subset = [s for s in scores if s.question_type == question_type]
        out[question_type] = {
            "n": float(len(subset)),
            f"recall_at_{k}": round(sum(s.hit_at(k) for s in subset) / len(subset), 4),
            "ndcg": round(sum(s.ndcg for s in subset) / len(subset), 4),
        }
    return out


def run(
    config: RunConfig,
    *,
    questions: list[GoldenQuestion] | None = None,
    corpus: EvalCorpus | None = None,
    cutoffs: list[int] | None = None,
    require_verified: float = 0.0,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    reranker: Reranker | None = None,
) -> RunResult:
    """Evaluate one configuration.

    `corpus` and `reranker` are injectable so a sweep pays for ingestion and
    the cross-encoder load once rather than once per arm. Left as None, each
    is built on demand — which is what a single `make eval` run wants.
    """
    questions = questions if questions is not None else load_golden_set()
    cutoffs = cutoffs or DEFAULT_CUTOFFS
    k = max(cutoffs)

    corpus = corpus or load_corpus(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        corpus_dir=corpus_dir,
    )
    retriever = corpus.build_retriever(
        config.strategy,
        candidate_k=config.candidate_k,
        rrf_k=config.rrf_k,
        reranker=reranker,
    )

    # Normalized once for the whole run: nDCG's ideal denominator scans every
    # chunk for every question, so re-normalizing per question would dominate.
    normalized_corpus = [normalize(text) for text in corpus.texts]

    scores = []
    for question in questions:
        retrieved = retriever.retrieve(question.question, k)
        scores.append(
            _score_question(
                question,
                [chunk.text for chunk in retrieved],
                [chunk.source_filename for chunk in retrieved],
                normalized_corpus,
                k,
            )
        )

    verified_fraction = sum(q.verified for q in questions) / len(questions) if questions else 0.0
    return RunResult(
        config=config,
        scores=aggregate(scores, k, cutoffs),
        per_question=scores,
        n_chunks=len(corpus.chunks),
        verified_fraction=verified_fraction,
        authoritative=verified_fraction >= require_verified > 0.0,
        by_type=_breakdown(scores, min(5, k)),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        platform=f"{platform.system()}-{platform.machine()}-py{platform.python_version()}",
    )


def format_summary(result: RunResult) -> str:
    s = result.scores
    lines = [
        f"config:   {result.config.label}  (candidate_k={result.config.candidate_k}, "
        f"rrf_k={result.config.rrf_k})",
        f"corpus:   {result.n_chunks} chunks",
        f"golden:   {s.n_questions} questions, {result.verified_fraction:.0%} human-verified",
        "",
        "  k   Recall@k   MRR@k",
    ]
    for cutoff in sorted(s.recall_at_k):
        lines.append(f" {cutoff:>2}     {s.recall_at_k[cutoff]:.3f}   {s.mrr_at_k[cutoff]:.3f}")
    lines += [f"\n nDCG@{s.k}: {s.ndcg_at_k:.3f}", "", " by question type (Recall@5 / nDCG):"]
    for question_type, row in result.by_type.items():
        lines.append(
            f"   {question_type:<12} n={int(row['n']):<3} "
            f"{row['recall_at_5']:.3f}   {row['ndcg']:.3f}"
        )
    if not result.authoritative:
        lines.append(
            "\n NOT AUTHORITATIVE — the golden set is not human-verified to the "
            "required threshold.\n Do not quote these numbers in the README."
        )
    return "\n".join(lines)


def write_result(result: RunResult, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"retrieval-{result.config.label}.json"
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval eval over the golden set.")
    parser.add_argument("--strategy", default="hybrid", choices=list(STRATEGIES))
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--candidate-k", type=int, default=settings.candidate_k)
    parser.add_argument("--rrf-k", type=int, default=settings.rrf_k)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--require-verified",
        type=float,
        default=0.0,
        help="fraction of the golden set that must be human-verified for the run "
        "to be marked authoritative (README numbers require 1.0)",
    )
    parser.add_argument("--no-write", action="store_true", help="print only, write no JSON")
    args = parser.parse_args(argv)

    config = RunConfig(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    result = run(
        config,
        questions=load_golden_set(args.questions),
        require_verified=args.require_verified,
        corpus_dir=args.corpus_dir,
    )
    print(format_summary(result))

    if not args.no_write:
        print(f"\nwrote {write_result(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
