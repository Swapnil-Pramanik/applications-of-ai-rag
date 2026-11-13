"""The adversarial set: questions the corpus cannot answer.

**Correct abstention needs negatives, and negatives are harder to build than
positives.** A golden question is verified by finding the supporting span. An
adversarial question is verified by establishing that *no* span anywhere in 18
papers supports it — proving a negative. That asymmetry is why these carry
`why_unanswerable` in writing: the claim being verified has to be stated before
it can be checked.

**`expected_catch` is the design point.** The abstention policy has two halves
that fail differently, so the set deliberately exercises both:

- `gate` — retrieval should return nothing plausible, so the confidence gate
  refuses before a generation call is made. Mostly out-of-domain and
  out-of-corpus questions.
- `model` — retrieval returns *genuinely relevant* chunks (the question names a
  paper we have) and only reading them reveals the answer isn't there. The gate
  cannot catch these by construction, and should not try: a threshold tuned
  high enough to reject "what did training the Transformer cost in dollars"
  would also reject every real question about the Transformer's training setup.

A set made only of `gate` cases would let a badly-tuned threshold look perfect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from eval.harness.golden_set import GOLDEN_SET_DIR

ADVERSARIAL_PATH = GOLDEN_SET_DIR / "adversarial.jsonl"

AdversarialType = Literal[
    "out_of_domain", "out_of_corpus", "absent_detail", "false_premise", "post_dating"
]
ExpectedCatch = Literal["gate", "model"]


@dataclass(frozen=True)
class AdversarialQuestion:
    id: str
    question: str
    type: AdversarialType
    why_unanswerable: str
    expected_catch: ExpectedCatch
    verified: bool = False
    notes: str = ""


def load_adversarial_set(path: Path = ADVERSARIAL_PATH) -> list[AdversarialQuestion]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        questions.append(AdversarialQuestion(**json.loads(line)))

    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in the adversarial set")
    return questions


def verified_fraction(questions: list[AdversarialQuestion]) -> float:
    return sum(q.verified for q in questions) / len(questions) if questions else 0.0
