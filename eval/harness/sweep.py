"""Day 9: sweeps chunk size and overlap, and picks the winner from the numbers.

**Two phases, not one grid.** A full cross product of 4 sizes x 3 overlaps is
12 ingestions of the whole corpus for a question that does not need it: chunk
size dominates, and overlap is second-order. So phase 1 sweeps size at a fixed
15% overlap, and phase 2 probes overlap (0% and 30%) only at whichever size
phase 1 selected. Six runs instead of twelve, and the one interaction worth
knowing about — does the best size want more or less overlap — still gets
measured.

**Every arm is committed, including the losers.** The original plan said
"pick a winner from data, and record the losing configs too": a sweep that
reports only its winner is indistinguishable from a guess, and the shape of
the curve either side of the winner is the actual evidence.

**Selection is on Recall@5, tie-broken by nDCG@10.** Recall@5 is what the
generation stage actually consumes (top-5 goes into the prompt), so it is the
metric with a downstream consequence. nDCG breaks ties because it is sensitive
to ordering *within* the cutoff, which Recall is not.

Overlap is expressed as a percentage of chunk size rather than an absolute
token count so the arms stay comparable: a fixed 150-character overlap is 38%
of a 400-char chunk and 10% of a 1500-char one, which would confound the very
variable phase 1 is trying to isolate.

Usage:  `make eval-sweep`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.retrieval.reranker import Reranker
from eval.harness.corpus_index import DEFAULT_CORPUS_DIR, STRATEGIES, load_corpus
from eval.harness.golden_set import QUESTIONS_PATH, load_golden_set
from eval.harness.runner import RESULTS_DIR, RunConfig, RunResult, run, write_result

# Phase 1: chunk size at a constant 15% overlap.
SIZE_ARMS = (400, 600, 1000, 1500)
BASE_OVERLAP_PCT = 0.15
# Phase 2: overlap fractions probed at the winning size.
OVERLAP_PCTS = (0.0, 0.30)

REPORT_PATH = RESULTS_DIR / "chunk-sweep.md"


def overlap_for(chunk_size: int, pct: float) -> int:
    return round(chunk_size * pct)


def _sort_key(result: RunResult) -> tuple[float, float]:
    return (result.scores.recall_at_k.get(5, 0.0), result.scores.ndcg_at_k)


def evaluate(
    chunk_size: int,
    chunk_overlap: int,
    *,
    strategy: str,
    reranker: Reranker | None,
    corpus_dir: Path,
    questions_path: Path,
) -> RunResult:
    """One arm: ingest at this chunking, retrieve, score, and write the JSON."""
    config = RunConfig(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    corpus = load_corpus(chunk_size=chunk_size, chunk_overlap=chunk_overlap, corpus_dir=corpus_dir)
    result = run(
        config,
        questions=load_golden_set(questions_path),
        corpus=corpus,
        require_verified=1.0,
        corpus_dir=corpus_dir,
        reranker=reranker,
    )
    write_result(result)
    return result


def format_report(
    phase1: list[RunResult], phase2: list[RunResult], winner: RunResult, strategy: str
) -> str:
    """The markdown table Day 9 owes the README."""

    def rows(results: list[RunResult]) -> list[str]:
        out = []
        for r in sorted(results, key=lambda r: (r.config.chunk_size, r.config.chunk_overlap)):
            s, by = r.scores, r.by_type
            mark = " **<-- winner**" if r.config.label == winner.config.label else ""
            out.append(
                f"| {r.config.chunk_size} | {r.config.chunk_overlap} "
                f"({r.config.chunk_overlap / r.config.chunk_size:.0%}) | {r.n_chunks} | "
                f"{s.recall_at_k.get(1, 0):.3f} | {s.recall_at_k.get(5, 0):.3f} | "
                f"{s.recall_at_k.get(10, 0):.3f} | {s.ndcg_at_k:.3f} | "
                + " / ".join(
                    f"{by[t]['recall_at_5']:.2f}"
                    for t in ("factual", "numeric", "comparative", "multi_hop")
                    if t in by
                )
                + f" |{mark}"
            )
        return out

    header = (
        "| chunk size | overlap | chunks | R@1 | R@5 | R@10 | nDCG@10 | "
        "fact / num / comp / multi-hop (R@5) |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Day 9 — chunk size and overlap sweep",
        "",
        f"Strategy held constant at `{strategy}` (the Day 8 winning arm), so every "
        "difference below is attributable to chunking alone. All arms scored over the "
        "same 60-question, 100%-human-verified golden set.",
        "",
        "Selection metric: **Recall@5**, tie-broken by nDCG@10 — top-5 is what the "
        "generation stage puts in the prompt, so it is the cutoff with a downstream "
        "consequence.",
        "",
        "## Phase 1 — chunk size at fixed 15% overlap",
        "",
        header,
        *rows(phase1),
        "",
        "## Phase 2 — overlap probed at the phase-1 winning size",
        "",
        header,
        *rows(phase2),
        "",
        "## Selected configuration",
        "",
        f"`chunk_size={winner.config.chunk_size}, chunk_overlap={winner.config.chunk_overlap}` "
        f"— Recall@5 {winner.scores.recall_at_k.get(5, 0):.3f}, "
        f"nDCG@10 {winner.scores.ndcg_at_k:.3f}, {winner.n_chunks} chunks.",
        "",
        "Per-type Recall@5 for the winner: "
        + ", ".join(
            f"{t} {row['recall_at_5']:.3f} (n={int(row['n'])})" for t, row in winner.by_type.items()
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep chunk size and overlap.")
    parser.add_argument("--strategy", default="hybrid+rerank", choices=list(STRATEGIES))
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args(argv)

    # Loaded once and threaded through every arm: the cross-encoder costs
    # seconds to construct and nothing about it varies across the sweep.
    reranker = Reranker() if args.strategy.endswith("+rerank") else None

    def arm(size: int, overlap: int) -> RunResult:
        result = evaluate(
            size,
            overlap,
            strategy=args.strategy,
            reranker=reranker,
            corpus_dir=args.corpus_dir,
            questions_path=args.questions,
        )
        print(
            f"  cs={size:<5} co={overlap:<4} chunks={result.n_chunks:<6} "
            f"R@5={result.scores.recall_at_k.get(5, 0):.3f}  "
            f"nDCG={result.scores.ndcg_at_k:.3f}"
        )
        return result

    print(f"phase 1 — chunk size at {BASE_OVERLAP_PCT:.0%} overlap  [{args.strategy}]")
    phase1 = [arm(size, overlap_for(size, BASE_OVERLAP_PCT)) for size in SIZE_ARMS]

    best_size = max(phase1, key=_sort_key).config.chunk_size
    print(f"\nphase 2 — overlap at cs={best_size}")
    seen = {overlap_for(best_size, BASE_OVERLAP_PCT)}
    phase2 = []
    for pct in OVERLAP_PCTS:
        overlap = overlap_for(best_size, pct)
        if overlap in seen:
            continue
        seen.add(overlap)
        phase2.append(arm(best_size, overlap))

    # The phase-1 arm at the winning size belongs in phase 2's table too: it is
    # the 15% point on the same overlap curve, and dropping it would leave the
    # comparison with a hole exactly where the incumbent sits.
    phase2.append(next(r for r in phase1 if r.config.chunk_size == best_size))

    winner = max(phase1 + phase2, key=_sort_key)
    REPORT_PATH.write_text(format_report(phase1, phase2, winner, args.strategy), encoding="utf-8")
    print(f"\nwinner: {winner.config.label}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
