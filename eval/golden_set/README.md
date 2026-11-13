# Golden set

The measuring instrument for every retrieval number this project reports. If a
label here is wrong, it does not look like a bug — it looks like a worse
retriever. Treat edits accordingly.

| File | Contents | Status |
|---|---|---|
| `questions.jsonl` | 60 answerable questions (factual, numeric, comparative, multi-hop) | drafted Day 5, **human-verified 60/60** |
| `adversarial.jsonl` | 40 unanswerable questions for the abstention gate | Day 11 |

## Schema

One JSON object per line:

```json
{
  "id": "g002",
  "question": "How many attention heads does the base Transformer use, and what is the dimension of each head?",
  "answer": "It uses h = 8 parallel attention heads, each of dimension d_k = d_v = d_model/h = 64.",
  "type": "numeric",
  "relevance": "any",
  "supporting_spans": [
    {
      "source_filename": "1706.03762.pdf",
      "page_number": 5,
      "span": "we employ h = 8 parallel attention layers, or heads"
    }
  ],
  "verified": false,
  "notes": ""
}
```

| Field | Meaning |
|---|---|
| `id` | `g001`…`g060`, stable — results files reference it |
| `type` | `factual` \| `numeric` \| `comparative` \| `multi_hop` |
| `relevance` | `any`: retrieving one span counts as a hit. `all`: every span must be retrieved (used for comparative/multi-hop questions, where a partial retrieval cannot support the answer) |
| `supporting_spans[].source_filename` | `<arxiv_id>.pdf`, matching `app/core/corpus.py` and the ingested filename |
| `supporting_spans[].page_number` | 1-indexed, as PyMuPDF reports it |
| `supporting_spans[].span` | verbatim from PyMuPDF's extraction, whitespace collapsed |
| `verified` | `true` only once a human has checked the answer and the citation against the PDF |

## Why spans and not chunk IDs

Chunk IDs are assigned by ingestion, so they change meaning whenever
`chunk_size` or `chunk_overlap` changes — and Day 9's ablation sweeps exactly
those parameters. Labelling by chunk ID would mean every sweep point grades
itself against a different answer key. A span is a property of the document, so
one label set scores every configuration fairly.

The second reason is circularity: the natural way to collect relevant chunk IDs
is to run the retriever and label what it returns, which quietly excludes
anything the retriever never surfaces and inflates Recall@k for free.

Scoring rule, implemented in `eval/harness/golden_set.py`: a retrieved chunk
supports a span if the normalized span is a substring of the normalized chunk
text. `normalize()` applies NFKC, rejoins words hyphenated across line breaks,
collapses whitespace and casefolds — to **both** sides, so what matters is that
the transformation is consistent, not that it is linguistically correct.

**Keep spans short** (roughly 5–15 words, hard cap 200 characters). A span that
straddles a chunk boundary belongs to neither chunk and scores as a miss even
when retrieval worked. At 150-character overlap, a short span can lose at most
one of the two neighbouring chunks.

## Verification protocol

Drafted by Claude from the extracted text, then human-verified **60/60**
against the canonical PDFs. "LLM-drafted, 100% human-verified against source"
is the honest answer if an interviewer asks — and the pass was not a rubber
stamp: it rejected one label (`g044`, below). The two halves:

**Machine half** — `make validate-golden` proves each span is verbatim on the
page and in the paper it claims, that ids are unique, that types and relevance
values are legal, that every paper is in the corpus, and that no span is long
enough to straddle a chunk boundary. It never sets `verified`.

**Human half** — for each question, open the cited page of the PDF and check:

1. the span really says what the answer claims (no inference the paper doesn't make);
2. the page number matches where you found it;
3. the question is answerable from the spans alone, without outside knowledge;
4. for `relevance: "all"`, that *both* spans are genuinely needed.

Then flip `verified` to `true` and put anything surprising in `notes`. Progress
shows in the validator's summary line (`human-verified: N/60`).

### ⚠ Verify against `data/papers/` — the canonical corpus

**This has already gone wrong once.** The first verification pass was done
against a hand-downloaded `research papers/<Title>.pdf` folder that used to be
tracked in git and was *not* the canonical corpus. Two of its files disagreed
with `data/papers/`:

| File | Problem |
|---|---|
| `research papers/BERT.pdf` | **A different paper entirely** — Koroteev's review *"BERT: A Review of Applications in NLP"*, not Devlin et al. |
| `research papers/Attention Is All You Need.pdf` | An 11-page version; the canonical arXiv copy is 15 pages |

72 of 80 spans sat on the same page in both copies, so most of that pass stood.
The 8 that didn't were re-checked against `data/papers/`. Seven passed. One,
`g044`, was **rejected** — see below.

That folder has since been deleted, so the trap is gone. This section stays
because it explains `g044`, and because it is the concrete argument for numeric
filenames.

### The one label the human pass caught: `g044`

Worth recording, because it is the class of error the machine validator
**structurally cannot** find. `g044` asked how BERT's pre-training objective
differs from the one used by *OpenAI GPT*, and its only span read:

> "which uses unidirectional language models for pre-training, BERT uses masked
> language models"

That span is verbatim on the page it claims, so `validate-golden` passed it.
But in the paper the sentence begins *"Unlike Radford et al. (2018)…"* — it
never says "OpenAI GPT". Answering the question as written required knowing
those are the same work, which is exactly the outside knowledge criterion 3
forbids. A span can be perfectly real and still not support its question.

Fixed by adding the p14 span that names both the system and its objective —
*"The most comparable existing pre-training method to BERT is OpenAI GPT, which
trains a left-to-right Transformer LM"* — and promoting the question to
`relevance: "all"`, matching the convention comparative questions follow. Both
spans were then human-checked and the question passed.

The canonical corpus is always `data/papers/<arxiv_id>.pdf`, fetched by
`make corpus`. Numeric filenames are the point: a label pointing at
`1810.04805.pdf` cannot silently start meaning a different document, which is
exactly the failure a title-named `BERT.pdf` allowed.

## Running it

```bash
make corpus            # fetch the 18 PDFs into data/papers/ (gitignored)
make validate-golden   # schema + every span against the real PDFs
```

CI runs the schema half only (`--schema-only`); it does not download the
corpus, and `tests/integration/test_golden_set_spans.py` skips there for the
same reason. The span check is a local gate.

---

## The adversarial set (`adversarial.jsonl`) — 40 questions, `human-verified 0/40`

Questions the corpus **cannot** answer. Correct abstention is a headline metric, and a metric
about refusing needs negatives to measure against.

### Why these are harder to verify than the answerable ones

A golden question is verified by *finding* its supporting span. An adversarial question is
verified by establishing that **no** span, anywhere in 18 papers, supports it — proving a
negative. There is no search that terminates in "absent". That asymmetry is why every row carries
a written `why_unanswerable`: the claim being checked has to be stated before it can be checked.

### `expected_catch` — the set deliberately exercises both halves of the policy

The abstention policy has two independent mechanisms that fail differently:

| `expected_catch` | n | Which mechanism should refuse | Typical shape |
|---|---|---|---|
| `gate` | 14 | The retrieval-side confidence gate, before any Gemini call | Out-of-domain, out-of-corpus |
| `model` | 26 | The model reading plausible chunks and judging them insufficient | Absent detail, false premise, post-dating |

The `model` cases are the interesting ones. "What did it cost in dollars to train the Transformer?"
retrieves the Transformer paper's training section — **genuinely the right chunks**, scoring high.
The gate cannot catch that and must not try: a threshold high enough to reject it would also
reject every real question about the Transformer's training setup. Only reading the chunks reveals
the paper reports FLOPs and GPU-days but never a dollar figure.

A set made only of `gate` cases would let a badly-tuned threshold look perfect.

### Question types

| type | n | What it tests |
|---|---|---|
| `out_of_corpus` | 10 | Real ML topics whose paper isn't among the 18 (Llama 2, Mamba, CLIP, Chinchilla…) |
| `absent_detail` | 10 | The right paper is in the corpus; it just never reports this fact |
| `false_premise` | 8 | Presupposes something the paper never did ("why did the Transformer authors choose LSTM decoders?") |
| `post_dating` | 7 | Asks about work published after the paper (PPO vs SAC, LoRA vs QLoRA) |
| `out_of_domain` | 5 | Not about ML at all — the floor case |

### Verification protocol — what the owner must do

For each of the 40, confirm the corpus genuinely cannot answer it, then set `verified: true`.

1. **Read `why_unanswerable` and decide whether you believe it.** These were drafted by Claude
   from recall of the papers and are **not** trustworthy until checked. The `absent_detail` rows
   are the riskiest — "the paper doesn't report X" is exactly the kind of claim that is wrong 10%
   of the time.
2. **Search the actual PDF** for the `absent_detail` and `false_premise` rows. Open
   `data/papers/<arxiv_id>.pdf` and search for the relevant term. If the paper *does* report it,
   the question is answerable and the row must be **rewritten or deleted**, not kept.
3. `out_of_domain` and `post_dating` rows are largely self-evident and can be checked quickly —
   the only real risk is a date error (confirm the cited work genuinely postdates the paper).
4. **A wrong label here corrupts the abstention rate in the flattering direction**: a question
   that is actually answerable, labelled unanswerable, scores as a correct abstention when the
   system refuses — rewarding the system for failing. This is the same failure mode as `g044` in
   the answerable set, and it is why `--require-verified` exists.

Until this is done, `make eval-calibrate` stamps `"authoritative": false` and the README may not
quote a catch rate.
