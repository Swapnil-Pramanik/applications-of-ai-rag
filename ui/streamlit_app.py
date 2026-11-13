"""Streamlit demo page. Drawing only: every decision lives in `ui/service.py`.

Not named `app.py` on purpose. Streamlit puts `ui/` on the import path, so a
`ui/app.py` would shadow the `app` package and break every import.

Run with `make demo` (which is `uv run python -m streamlit run ui/streamlit_app.py`).
`python -m` puts the repo root on the import path, so `app`, `eval` and `ui`
import normally.
"""

from __future__ import annotations

import streamlit as st

from ui.service import (
    EXAMPLES,
    STRATEGY_LABELS,
    DemoResult,
    DemoService,
    load_service,
    numbered_claims,
    paper_title,
)

st.set_page_config(page_title="Ask the Papers", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading 18 papers and the retrieval models (first run is slow)…")
def get_service() -> DemoService:
    # Cached across reruns and browser tabs: the corpus and both models load once.
    return load_service()


def use_example(question: str) -> None:
    st.session_state["question"] = question
    st.session_state["run"] = True


# --- sidebar: settings and how it works ------------------------------------

service = get_service()

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox(
        "Retriever",
        options=list(STRATEGY_LABELS),
        format_func=STRATEGY_LABELS.__getitem__,
        help="Switch to compare the six arms of the ablation on the same question.",
    )
    generate = st.toggle(
        "Generate an answer with Gemini",
        value=True,
        help="Off = show retrieval only. Uses no API quota.",
    )

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Dense** search (BGE embeddings + FAISS) and **BM25** keyword search each "
        "find 20 candidate passages.\n"
        "2. **Reciprocal Rank Fusion** merges the two lists.\n"
        "3. A **cross-encoder** rescores them and keeps the top 5.\n"
        "4. **Confidence gate**: if no passage scores above the threshold, the system "
        "refuses without calling the LLM.\n"
        "5. **Gemini** answers using only those passages and cites each claim; "
        "invented or missing citations make it refuse."
    )

    st.divider()
    with st.expander(f"Corpus: {len(service.papers())} papers, {service.chunk_count:,} chunks"):
        for title, count in service.papers():
            st.markdown(f"- {title} ({count})")


# --- main column -----------------------------------------------------------

st.title("Ask the Papers")
st.caption(
    "Questions over 18 landmark ML papers. Every claim cites a paper, page and passage, "
    "and the system says so when the papers don't contain the answer."
)

st.markdown("**Try an example:**")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES, strict=True):
    column.button(
        example.label,
        help=example.question,
        on_click=use_example,
        args=(example.question,),
        use_container_width=True,
    )

with st.form("ask"):
    question = st.text_input(
        "Your question", key="question", placeholder="e.g. How does LoRA initialise A and B?"
    )
    submitted = st.form_submit_button("Ask", type="primary")

if (submitted or st.session_state.pop("run", False)) and question.strip():
    with st.spinner("Retrieving and answering…"):
        st.session_state["result"] = service.ask(question.strip(), strategy, generate=generate)


def render_outcome(result: DemoResult) -> None:
    status = result.status
    if status == "error":
        st.error(result.error)
    elif status == "refused_by_gate":
        assert result.gate is not None
        st.warning(
            "**Refused before calling the LLM.** No retrieved passage looked relevant enough "
            f"({result.gate.reason})."
        )
    elif status == "refused_by_model":
        assert result.answer is not None
        st.warning(
            "**The model read the top passages and judged them insufficient.** "
            f"{result.answer.abstain_reason}"
        )
    elif status == "retrieval_only":
        st.info("Generation is off. Showing what the retriever found.")

    if result.answer is not None and result.answer.answered:
        st.subheader("Answer")
        for claim, numbers in numbered_claims(result.answer):
            marks = "".join(f"[{n}]" for n in numbers)
            st.markdown(f"{claim.text} **{marks}**")

        st.markdown("**Sources**")
        for n, citation in enumerate(result.answer.citations, start=1):
            title = paper_title(citation.source_filename)
            with st.expander(f"[{n}] {title}, page {citation.page_number}"):
                st.markdown(f"> {citation.quote}")


def render_retrieved(result: DemoResult) -> None:
    st.subheader(f"Top {len(result.retrieved)} retrieved passages")
    if result.gate is None:
        st.caption(
            f"{STRATEGY_LABELS[result.strategy]}: scores are not cross-encoder scores, "
            "so the confidence gate is off for this retriever."
        )
    else:
        st.caption(
            f"Cross-encoder scores. Gate threshold: {result.gate.threshold:.1f} "
            "(passages below it are not shown to the LLM)."
        )

    for rank, chunk in enumerate(result.retrieved, start=1):
        passed = result.passed_gate(chunk)
        if passed is None:
            mark = ""
        else:
            mark = " :green[✓ passed gate]" if passed else " :red[✗ below gate]"
        label = (
            f"#{rank} · {paper_title(chunk.source_filename)} · page {chunk.page_number} · "
            f"score {chunk.score:.3f}{mark}"
        )
        with st.expander(label):
            st.write(chunk.text)


result: DemoResult | None = st.session_state.get("result")
if result is not None:
    st.divider()
    st.markdown(f"**Q:** {result.question}")
    answer_col, evidence_col = st.columns([3, 2], gap="large")
    with answer_col:
        render_outcome(result)
    with evidence_col:
        render_retrieved(result)
    st.caption(f"{STRATEGY_LABELS[result.strategy]} · {result.seconds:.2f}s")
