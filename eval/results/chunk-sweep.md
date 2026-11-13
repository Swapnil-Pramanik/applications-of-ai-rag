# Day 9 — chunk size and overlap sweep

Strategy held constant at `hybrid+rerank` (the Day 8 winning arm), so every difference below is attributable to chunking alone. All arms scored over the same 60-question, 100%-human-verified golden set.

Selection metric: **Recall@5**, tie-broken by nDCG@10 — top-5 is what the generation stage puts in the prompt, so it is the cutoff with a downstream consequence.

## Phase 1 — chunk size at fixed 15% overlap

| chunk size | overlap | chunks | R@1 | R@5 | R@10 | nDCG@10 | fact / num / comp / multi-hop (R@5) |
|---|---|---|---|---|---|---|---|
| 400 | 60 (15%) | 3423 | 0.250 | 0.433 | 0.550 | 0.454 | 0.41 / 0.68 / 0.42 / 0.08 |
| 600 | 90 (15%) | 2277 | 0.317 | 0.567 | 0.650 | 0.539 | 0.65 / 0.74 / 0.58 / 0.17 | **<-- winner**
| 1000 | 150 (15%) | 1446 | 0.250 | 0.567 | 0.633 | 0.487 | 0.65 / 0.79 / 0.58 / 0.08 |
| 1500 | 225 (15%) | 1002 | 0.167 | 0.450 | 0.633 | 0.434 | 0.59 / 0.63 / 0.42 / 0.00 |

## Phase 2 — overlap probed at the phase-1 winning size

| chunk size | overlap | chunks | R@1 | R@5 | R@10 | nDCG@10 | fact / num / comp / multi-hop (R@5) |
|---|---|---|---|---|---|---|---|
| 600 | 0 (0%) | 2137 | 0.250 | 0.417 | 0.533 | 0.437 | 0.47 / 0.58 / 0.42 / 0.08 |
| 600 | 90 (15%) | 2277 | 0.317 | 0.567 | 0.650 | 0.539 | 0.65 / 0.74 / 0.58 / 0.17 | **<-- winner**
| 600 | 180 (30%) | 2697 | 0.300 | 0.517 | 0.617 | 0.499 | 0.65 / 0.68 / 0.50 / 0.08 |

## Selected configuration

`chunk_size=600, chunk_overlap=90` — Recall@5 0.567, nDCG@10 0.539, 2277 chunks.

Per-type Recall@5 for the winner: comparative 0.583 (n=12), factual 0.647 (n=17), multi_hop 0.167 (n=12), numeric 0.737 (n=19).
