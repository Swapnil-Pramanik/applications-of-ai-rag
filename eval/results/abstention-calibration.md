# Abstention gate calibration (Day 11)

Swept over 60 answerable golden questions and 40 adversarial ones (14 of which the gate is expected to catch; the rest are the model-side half's job by design).

Retrieval is `hybrid+rerank`, so scores are cross-encoder logits. **No Gemini calls are made** — the gate's decision depends only on retrieval scores, which is what makes sweeping it free.

| threshold | answerable pass | gate catch | all-adversarial catch | model-expected caught early | Youden J |
|---|---|---|---|---|---|
| -10.0 | 1.000 | 0.143 | 0.050 | 0.000 | +0.143 |
| -8.0 | 1.000 | 0.357 | 0.125 | 0.000 | +0.357 |
| -6.0 | 1.000 | 0.571 | 0.200 | 0.000 | +0.571 |
| -4.0 | 1.000 | 0.786 | 0.300 | 0.038 | +0.786 |
| -2.0 | 0.950 | 0.929 | 0.425 | 0.154 | +0.879 | **<-- selected**
| -1.0 | 0.917 | 0.929 | 0.500 | 0.269 | +0.845 |
| +0.0 | 0.900 | 0.929 | 0.575 | 0.385 | +0.829 |
| +1.0 | 0.850 | 0.929 | 0.650 | 0.500 | +0.779 |
| +2.0 | 0.783 | 0.929 | 0.725 | 0.615 | +0.712 |
| +4.0 | 0.617 | 1.000 | 0.900 | 0.846 | +0.617 |
| +6.0 | 0.300 | 1.000 | 0.975 | 0.962 | +0.300 |
| +8.0 | 0.067 | 1.000 | 1.000 | 1.000 | +0.067 |

**Selected threshold: -2.0** — lets through 95.0% of answerable questions while refusing 92.9% of the questions it is meant to refuse.

Reading the columns: *answerable pass* is the one to protect — a gate that refuses real questions is a broken product. *gate catch* is the gate's actual job. *model-expected caught early* is a bonus rather than a target: those questions retrieve genuinely relevant chunks and were always going to be the model's to refuse.

> ⚠️ **NOT AUTHORITATIVE.** The adversarial set is not yet human-verified, so these numbers describe the labels as much as the gate. Do not quote them in the README.
