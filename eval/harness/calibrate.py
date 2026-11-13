"""Sweeps the confidence threshold and reports the trade. No Gemini calls.

**This measures the gate alone, on purpose.** Running the whole pipeline would
need a generation call per question per threshold — 40+60 questions across a
dozen thresholds is thousands of calls, and the free tier does not have them.
The gate's decision depends only on cross-encoder logits, so the entire sweep
is retrieval-side and costs nothing but CPU. The model-side half of the policy
is measured separately, once, at the chosen threshold.

**The two error types are not symmetric and the report keeps them apart.**

- A **false abstention** is refusing a question the corpus *can* answer. It is
  the expensive error: the user sees a system that doesn't work.
- A **missed catch** is letting an unanswerable question through the gate. It
  is only half an error, because the model-side abstention is still downstream
  and is specifically meant to catch the `expected_catch: "model"` cases.

So a threshold is not chosen by maximising raw accuracy over the pooled sets —
that would reward strangling recall to catch questions the gate was never meant
to catch. It is chosen on the gate-catchable subset, subject to keeping false
abstentions low. `Youden's J` on that subset is reported alongside the raw
rates so the choice can be argued with rather than taken on faith.

Usage:  `make eval-calibrate`
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings
from app.retrieval.confidence import ConfidenceGate
from app.retrieval.reranker import Reranker
from eval.harness.adversarial import (
    ADVERSARIAL_PATH,
    AdversarialQuestion,
    load_adversarial_set,
)
from eval.harness.adversarial import verified_fraction as adversarial_verified
from eval.harness.corpus_index import DEFAULT_CORPUS_DIR, load_corpus
from eval.harness.golden_set import QUESTIONS_PATH, GoldenQuestion, load_golden_set
from eval.harness.runner import RESULTS_DIR

DEFAULT_THRESHOLDS = [-10.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
REPORT_PATH = RESULTS_DIR / "abstention-calibration.md"


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    # Answerable questions the gate let through. Higher is better.
    answerable_pass_rate: float
    # Gate-catchable adversarials the gate refused. Higher is better.
    gate_catch_rate: float
    # Adversarials the gate refused, over all 40 — context, not a target.
    overall_catch_rate: float
    # Model-expected adversarials the gate wrongly refused. Not an error, but
    # it means the gate is doing work the model was going to do anyway.
    model_expected_caught_early: float
    youden_j: float


def top_scores(questions: list[str], retriever_retrieve, k: int) -> list[list[float]]:  # type: ignore[no-untyped-def]
    """Retrieve once per question and keep the scores. The sweep reuses these.

    Retrieval is the expensive part and it does not depend on the threshold, so
    running it once per question instead of once per (question, threshold) is
    the difference between a minute and a quarter of an hour.
    """
    return [[chunk.score for chunk in retriever_retrieve(q, k)] for q in questions]


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def sweep(
    answerable_scores: list[list[float]],
    adversarial_scores: list[list[float]],
    adversarial: list[AdversarialQuestion],
    thresholds: list[float],
    *,
    min_supporting_chunks: int = 1,
) -> list[ThresholdRow]:
    rows = []
    gate_idx = [i for i, q in enumerate(adversarial) if q.expected_catch == "gate"]
    model_idx = [i for i, q in enumerate(adversarial) if q.expected_catch == "model"]

    for threshold in thresholds:
        gate = ConfidenceGate(threshold=threshold, min_supporting_chunks=min_supporting_chunks)

        # `gate.admits` is the same call the service makes — see
        # `ConfidenceGate.admits`. Reimplementing it here would eventually
        # calibrate a gate production does not run.
        answerable_passed = sum(gate.admits(s) for s in answerable_scores)
        adversarial_caught = [not gate.admits(s) for s in adversarial_scores]

        answerable_pass_rate = _rate(answerable_passed, len(answerable_scores))
        gate_catch_rate = _rate(sum(adversarial_caught[i] for i in gate_idx), len(gate_idx))
        rows.append(
            ThresholdRow(
                threshold=threshold,
                answerable_pass_rate=answerable_pass_rate,
                gate_catch_rate=gate_catch_rate,
                overall_catch_rate=_rate(sum(adversarial_caught), len(adversarial_caught)),
                model_expected_caught_early=_rate(
                    sum(adversarial_caught[i] for i in model_idx), len(model_idx)
                ),
                # TPR on the gate-catchable subset minus FPR on answerable
                # questions. Both measured against what the gate is *for*.
                youden_j=round(gate_catch_rate - (1.0 - answerable_pass_rate), 4),
            )
        )
    return rows


def select(rows: list[ThresholdRow], *, min_answerable_pass_rate: float) -> ThresholdRow:
    """Best Youden's J among thresholds that keep enough real questions alive.

    The floor is a hard constraint rather than part of the objective: a gate
    that refuses real questions is worse than one that leans on the model-side
    abstention, whatever its catch rate looks like.
    """
    eligible = [r for r in rows if r.answerable_pass_rate >= min_answerable_pass_rate]
    if not eligible:
        # Reporting the least-bad row and saying so beats silently returning
        # a threshold that fails the constraint everywhere.
        return max(rows, key=lambda r: r.answerable_pass_rate)
    return max(eligible, key=lambda r: (r.youden_j, r.answerable_pass_rate))


def format_report(
    rows: list[ThresholdRow],
    chosen: ThresholdRow,
    *,
    n_answerable: int,
    adversarial: list[AdversarialQuestion],
    authoritative: bool,
) -> str:
    n_gate = sum(q.expected_catch == "gate" for q in adversarial)
    lines = [
        "# Abstention gate calibration (Day 11)",
        "",
        f"Swept over {n_answerable} answerable golden questions and {len(adversarial)} "
        f"adversarial ones ({n_gate} of which the gate is expected to catch; the rest are "
        "the model-side half's job by design).",
        "",
        "Retrieval is `hybrid+rerank`, so scores are cross-encoder logits. **No Gemini calls "
        "are made** — the gate's decision depends only on retrieval scores, which is what makes "
        "sweeping it free.",
        "",
        "| threshold | answerable pass | gate catch | all-adversarial catch "
        "| model-expected caught early | Youden J |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = " **<-- selected**" if row.threshold == chosen.threshold else ""
        lines.append(
            f"| {row.threshold:+.1f} | {row.answerable_pass_rate:.3f} | "
            f"{row.gate_catch_rate:.3f} | {row.overall_catch_rate:.3f} | "
            f"{row.model_expected_caught_early:.3f} | {row.youden_j:+.3f} |{mark}"
        )
    lines += [
        "",
        f"**Selected threshold: {chosen.threshold:+.1f}** — lets through "
        f"{chosen.answerable_pass_rate:.1%} of answerable questions while refusing "
        f"{chosen.gate_catch_rate:.1%} of the questions it is meant to refuse.",
        "",
        "Reading the columns: *answerable pass* is the one to protect — a gate that refuses real "
        "questions is a broken product. *gate catch* is the gate's actual job. *model-expected "
        "caught early* is a bonus rather than a target: those questions retrieve genuinely "
        "relevant chunks and were always going to be the model's to refuse.",
    ]
    if not authoritative:
        lines += [
            "",
            "> ⚠️ **NOT AUTHORITATIVE.** The adversarial set is not yet human-verified, so these "
            "numbers describe the labels as much as the gate. Do not quote them in the README.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the abstention threshold.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--adversarial", type=Path, default=ADVERSARIAL_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--min-answerable-pass-rate", type=float, default=0.95)
    parser.add_argument("--require-verified", type=float, default=0.0)
    args = parser.parse_args(argv)

    golden: list[GoldenQuestion] = load_golden_set(args.questions)
    adversarial = load_adversarial_set(args.adversarial)

    corpus = load_corpus(corpus_dir=args.corpus_dir)
    retriever = corpus.build_retriever("hybrid+rerank", reranker=Reranker())

    print(f"scoring {len(golden)} answerable + {len(adversarial)} adversarial questions...")
    answerable_scores = top_scores([q.question for q in golden], retriever.retrieve, settings.top_k)
    adversarial_scores = top_scores(
        [q.question for q in adversarial], retriever.retrieve, settings.top_k
    )

    rows = sweep(answerable_scores, adversarial_scores, adversarial, DEFAULT_THRESHOLDS)
    chosen = select(rows, min_answerable_pass_rate=args.min_answerable_pass_rate)

    verified = adversarial_verified(adversarial)
    authoritative = verified >= args.require_verified > 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        format_report(
            rows,
            chosen,
            n_answerable=len(golden),
            adversarial=adversarial,
            authoritative=authoritative,
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "abstention-calibration.json").write_text(
        json.dumps(
            {
                "rows": [asdict(r) for r in rows],
                "selected": asdict(chosen),
                "min_answerable_pass_rate": args.min_answerable_pass_rate,
                "adversarial_verified_fraction": round(verified, 4),
                "authoritative": authoritative,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "platform": f"{platform.system()}-{platform.machine()}",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for row in rows:
        mark = "  <-- selected" if row.threshold == chosen.threshold else ""
        print(
            f"  t={row.threshold:+5.1f}  answerable_pass={row.answerable_pass_rate:.3f}  "
            f"gate_catch={row.gate_catch_rate:.3f}  J={row.youden_j:+.3f}{mark}"
        )
    print(f"\nselected threshold: {chosen.threshold:+.1f}")
    if not authoritative:
        print("NOT AUTHORITATIVE — adversarial set is not human-verified. Do not quote.")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
