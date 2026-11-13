import type { QueryResponse } from "@/lib/api";

type State = "idle" | "running" | "done" | "skipped" | "stopped" | "refused" | "error";

interface Stage {
  name: string;
  detail: string;
  state: State;
  note?: string;
}

/** Which of the five stages ran for this question, and where it stopped. */
export function stagesFor(
  strategy: string,
  generate: boolean,
  result: QueryResponse | null,
  loading: boolean,
): Stage[] {
  const [base, suffix] = strategy.split("+");
  const reranked = suffix === "rerank";
  const ran = (on: boolean): State => (loading ? (on ? "running" : "skipped") : result ? (on ? "done" : "skipped") : on ? "idle" : "skipped");

  const retrieveDetail =
    base === "hybrid" ? "Dense (BGE + FAISS) and BM25, top 20 each" : base === "dense" ? "Dense only (BGE + FAISS)" : "BM25 keyword search only";

  let gate: State = ran(reranked);
  let answer: State = ran(generate);
  let gateNote: string | undefined;
  let answerNote: string | undefined;
  if (result && !loading) {
    if (result.status === "refused_by_gate") {
      gate = "stopped";
      answer = "skipped";
      answerNote = "never called";
    } else if (result.status === "refused_by_model") {
      answer = "refused";
      answerNote = "refused";
    } else if (result.status === "error") {
      answer = "error";
      answerNote = "unavailable";
    }
    if (result.gate?.top_score != null) gateNote = `best ${result.gate.top_score.toFixed(2)}`;
  }
  if (!generate) answerNote = "off";

  return [
    { name: "Retrieve", detail: retrieveDetail, state: ran(true) },
    { name: "Fuse", detail: "Reciprocal Rank Fusion merges the two lists", state: ran(base === "hybrid") },
    { name: "Rerank", detail: "Cross-encoder rescores, keeps top 5", state: ran(reranked) },
    { name: "Gate", detail: "Refuse if no passage scores ≥ −2.0", state: gate, note: gateNote },
    { name: "Answer", detail: "Gemini answers from the passages, citing each claim", state: answer, note: answerNote },
  ];
}

const STYLE: Record<State, string> = {
  idle: "border-line bg-card",
  running: "border-accent bg-accent-soft animate-pulse",
  done: "border-good/50 bg-good-soft",
  skipped: "border-dashed border-line bg-transparent opacity-50",
  stopped: "border-warn bg-warn-soft",
  refused: "border-warn bg-warn-soft",
  error: "border-bad bg-bad-soft",
};

const MARK: Record<State, string> = {
  idle: "",
  running: "…",
  done: "✓",
  skipped: "–",
  stopped: "✋",
  refused: "✋",
  error: "!",
};

export default function Pipeline({ stages }: { stages: Stage[] }) {
  return (
    <ol className="grid grid-cols-2 gap-2 sm:grid-cols-5" aria-label="Pipeline stages">
      {stages.map((stage, i) => (
        <li
          key={stage.name}
          className={`relative rounded-xl border p-3 transition-colors duration-300 ${STYLE[stage.state]}`}
        >
          <div className="flex items-center justify-between text-xs font-medium text-muted">
            <span>Step {i + 1}</span>
            <span aria-hidden className="text-sm text-ink">{MARK[stage.state]}</span>
          </div>
          <div className="mt-1 font-semibold">{stage.name}</div>
          <p className="mt-1 text-xs leading-snug text-muted">{stage.detail}</p>
          {stage.note && <p className="mt-1 text-xs font-medium text-ink">{stage.note}</p>}
        </li>
      ))}
    </ol>
  );
}
