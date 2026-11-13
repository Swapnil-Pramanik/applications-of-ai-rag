"""Retrieval metrics computed against span-labelled golden questions.

**The unit of relevance here is a question, not a chunk.** The golden set says
"this question is answered by these spans" (see `golden_set`), and a span sits
inside whichever chunk the current chunking configuration happens to produce.
So every metric below is derived from one primitive — `first_satisfied_rank`,
the smallest prefix of the ranked list that contains the support the question
requires. That single definition handles both relevance modes correctly:
`any` is satisfied as soon as one span shows up, `all` only once the last of
them does. Defining recall directly on chunks instead would silently score a
multi-hop question as a hit the moment half its evidence arrived.

**Why nDCG needs the whole corpus.** Recall and MRR only look at the ranked
list, but nDCG divides by the *ideal* ranking, which requires knowing how many
relevant chunks exist at all — a question whose evidence lives in one chunk
cannot do better than one hit at rank 1, while a question with six relevant
chunks is penalized for surfacing only two. `relevant_chunk_count` scans the
ingested corpus once to get that denominator. It is the expensive call in this
module, which is why the runner computes it a single time per configuration
and passes it in rather than letting each metric recompute it.

**Normalization is the hot path.** Every comparison runs `normalize()` on both
sides. Chunk text is normalized once per corpus and reused across all 60
questions; without that the scan is thousands of redundant NFKC passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from eval.harness.golden_set import GoldenQuestion, normalize


def first_satisfied_rank(question: GoldenQuestion, texts: list[str]) -> int | None:
    """1-based rank of the shortest prefix of `texts` that satisfies `question`.

    Returns None if the full list never satisfies it. Evaluated prefix by
    prefix rather than by locating individual spans because `relevance="all"`
    is only satisfied by the *last* required span to arrive, and that is a
    property of the prefix, not of any one chunk.
    """
    for rank in range(1, len(texts) + 1):
        if question.is_satisfied_by(texts[:rank]):
            return rank
    return None


def is_relevant_chunk(question: GoldenQuestion, normalized_text: str) -> bool:
    """Does this chunk carry at least one of the question's supporting spans?

    Takes text that is *already* normalized — the caller normalizes the corpus
    once and reuses it, so re-normalizing here would dominate the runtime.
    """
    return any(normalize(span.span) in normalized_text for span in question.supporting_spans)


def relevant_chunk_count(question: GoldenQuestion, normalized_corpus: list[str]) -> int:
    """How many chunks in the whole corpus could have answered this question.

    This is nDCG's ideal-ranking denominator. It is configuration-dependent on
    purpose: a smaller `chunk_size` splits the same span-bearing text across
    more chunks, so the ideal ranking changes with it, and the Day 9 sweep is
    entitled to see that.
    """
    return sum(is_relevant_chunk(question, text) for text in normalized_corpus)


def dcg(gains: list[float]) -> float:
    return sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(
    question: GoldenQuestion,
    normalized_retrieved: list[str],
    k: int,
    n_relevant: int,
) -> float:
    """Binary-gain nDCG@k. Zero when the corpus holds no relevant chunk at all.

    A question with no relevant chunk anywhere in the corpus is a labelling or
    ingestion failure, not a retrieval failure — but scoring it 0 rather than
    skipping it keeps the denominator of every reported average equal to the
    number of questions, so two runs over the same golden set are always
    comparable.
    """
    if n_relevant == 0:
        return 0.0

    gains = [float(is_relevant_chunk(question, text)) for text in normalized_retrieved[:k]]
    ideal = dcg([1.0] * min(n_relevant, k))
    return dcg(gains) / ideal if ideal else 0.0


@dataclass(frozen=True)
class QuestionScore:
    """Per-question detail, kept so a run can be debugged, not just averaged."""

    question_id: str
    question_type: str
    rank: int | None  # first_satisfied_rank over the full retrieved list
    ndcg: float
    n_relevant: int
    retrieved_sources: tuple[str, ...]

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    def reciprocal_rank(self, k: int) -> float:
        # Narrowed against `rank` directly rather than via `hit_at`, which
        # mypy cannot see through to prove the divisor is not None.
        return 1.0 / self.rank if self.rank is not None and self.rank <= k else 0.0


@dataclass(frozen=True)
class AggregateScores:
    n_questions: int
    recall_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    ndcg_at_k: float
    k: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_questions": self.n_questions,
            "k": self.k,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "mrr_at_k": {str(k): round(v, 4) for k, v in self.mrr_at_k.items()},
            "ndcg_at_k": round(self.ndcg_at_k, 4),
        }


def aggregate(scores: list[QuestionScore], k: int, cutoffs: list[int]) -> AggregateScores:
    """Mean the per-question scores. Empty input yields zeros, never a crash."""
    n = len(scores)
    if n == 0:
        return AggregateScores(0, {c: 0.0 for c in cutoffs}, {c: 0.0 for c in cutoffs}, 0.0, k)

    return AggregateScores(
        n_questions=n,
        recall_at_k={c: sum(s.hit_at(c) for s in scores) / n for c in cutoffs},
        mrr_at_k={c: sum(s.reciprocal_rank(c) for s in scores) / n for c in cutoffs},
        ndcg_at_k=sum(s.ndcg for s in scores) / n,
        k=k,
    )
