import type { QueryResponse } from "@/lib/api";

export type StageState = "idle" | "running" | "done" | "skipped" | "stopped" | "error";

export interface Stage {
  name: string;
  detail: string;
  state: StageState;
  note?: string;
}

/** Which of the five pipeline steps ran for one question, and where it stopped. */
export function stagesFor(
  strategy: string,
  generate: boolean,
  result: QueryResponse | null,
  running: boolean,
): Stage[] {
  const [base, suffix] = strategy.split("+");
  const reranked = suffix === "rerank";
  const ran = (on: boolean): StageState =>
    !on ? "skipped" : running ? "running" : result ? "done" : "idle";

  let gate = ran(reranked);
  let answer = ran(generate);
  let gateNote: string | undefined;
  let answerNote: string | undefined = generate ? undefined : "off";
  if (result && !running) {
    if (result.status === "refused_by_gate") {
      gate = "stopped";
      answer = "skipped";
      answerNote = "never called";
    } else if (result.status === "refused_by_model") {
      answer = "stopped";
      answerNote = "model refused";
    } else if (result.status === "error") {
      answer = "error";
      answerNote = "unavailable";
    }
    if (result.gate?.top_score != null) gateNote = `best ${result.gate.top_score.toFixed(2)}`;
  }

  return [
    {
      name: "Retrieve",
      detail:
        base === "hybrid"
          ? "Dense (BGE + FAISS) and BM25 keyword search, top 20 each"
          : base === "dense"
            ? "Dense search only (BGE + FAISS)"
            : "BM25 keyword search only",
      state: ran(true),
    },
    { name: "Fuse", detail: "Reciprocal Rank Fusion merges the two lists", state: ran(base === "hybrid") },
    { name: "Rerank", detail: "A cross-encoder rescores and keeps the top 5", state: ran(reranked) },
    { name: "Gate", detail: "Refuse if no passage scores above the threshold", state: gate, note: gateNote },
    { name: "Answer", detail: "The LLM answers from the passages and cites each claim", state: answer, note: answerNote },
  ];
}

const PILL: Record<StageState, string> = {
  idle: "border-line text-muted",
  running: "border-accent/40 bg-accent-soft text-accent animate-pulse",
  done: "border-good/30 bg-good-soft text-good",
  skipped: "border-dashed border-line text-muted/60 line-through",
  stopped: "border-warn/40 bg-warn-soft text-warn",
  error: "border-bad/40 bg-bad-soft text-bad",
};

const MARK: Record<StageState, string> = {
  idle: "",
  running: "",
  done: "✓ ",
  skipped: "",
  stopped: "✋ ",
  error: "! ",
};

/** One row of five small pills; hover a pill for what that step does. */
export default function Pipeline({ stages }: { stages: Stage[] }) {
  return (
    <ol className="flex flex-wrap items-center gap-1 text-xs" aria-label="Pipeline steps">
      {stages.map((stage, i) => (
        <li key={stage.name} className="flex items-center gap-1">
          <span
            title={`${stage.detail}${stage.state === "skipped" ? " (not used by this retriever)" : ""}`}
            className={`rounded-full border px-2 py-0.5 font-medium ${PILL[stage.state]}`}
          >
            {MARK[stage.state]}
            {stage.name}
            {stage.note && <span className="font-normal opacity-80"> · {stage.note}</span>}
          </span>
          {i < stages.length - 1 && <span className="text-muted/50">→</span>}
        </li>
      ))}
    </ol>
  );
}
