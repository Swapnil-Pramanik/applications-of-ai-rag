"""Every golden-set span must really be on the page it claims, in the real PDF.

Skipped when `data/papers/` is absent (CI does not download the corpus), which
is why `make validate-golden` exists as the local gate — see eval/golden_set/README.md.
"""

from __future__ import annotations

import pytest

from eval.harness.golden_set import load_golden_set
from eval.harness.validate_golden import DEFAULT_CORPUS_DIR, _check_spans

pytestmark = pytest.mark.skipif(
    not DEFAULT_CORPUS_DIR.exists(),
    reason="corpus PDFs not present — run `make corpus`",
)


def test_every_span_is_verbatim_on_its_stated_page() -> None:
    assert _check_spans(load_golden_set(), DEFAULT_CORPUS_DIR) == []


def test_a_tampered_span_is_caught() -> None:
    """Guards the guard: the check must actually fail on a wrong label."""
    question = load_golden_set()[0]
    tampered = [
        type(question)(
            **{
                **question.__dict__,
                "supporting_spans": (
                    type(question.supporting_spans[0])(
                        source_filename=question.supporting_spans[0].source_filename,
                        page_number=question.supporting_spans[0].page_number,
                        span="a sentence that appears in no paper in this corpus",
                    ),
                ),
            }
        )
    ]
    assert _check_spans(tampered, DEFAULT_CORPUS_DIR) != []
