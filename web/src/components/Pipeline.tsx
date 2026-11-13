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
    } else if (result.status === "refused_by_model") {
      answer = "stopped";
      answerNote = "refused";
    } else if (result.status === "error") {
      answer = "error";
    }
    if (result.gate?.top_score != null) gateNote = result.gate.top_score.toFixed(2);
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

const TONE: Record<StageState, string> = {
  idle: "text-faint",
  running: "text-accent animate-pulse",
  done: "text-muted",
  skipped: "text-faint/70 line-through",
  stopped: "text-warn",
  error: "text-bad",
};

const MARK: Record<StageState, string> = {
  idle: "○",
  running: "○",
  done: "✓",
  skipped: "–",
  stopped: "■",
  error: "!",
};

/** One quiet line: ✓ Retrieve ✓ Fuse ✓ Rerank ✓ Gate 8.93 ✓ Answer. Hover a step for detail. */
export default function Pipeline({ stages }: { stages: Stage[] }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs" aria-label="Pipeline steps">
      {stages.map((stage) => (
        <li
          key={stage.name}
          className={TONE[stage.state]}
          title={`${stage.detail}${stage.state === "skipped" ? " (not used by this retriever)" : ""}`}
        >
          <span aria-hidden className="mr-1">{MARK[stage.state]}</span>
          {stage.name}
          {stage.note && <span className="ml-1 tabular-nums opacity-80">{stage.note}</span>}
        </li>
      ))}
    </ol>
  );
}
