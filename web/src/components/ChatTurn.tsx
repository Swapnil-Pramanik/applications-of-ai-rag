"use client";

import { ChevronRight, CircleSlash, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";

import Evidence from "@/components/Evidence";
import Pipeline, { stagesFor } from "@/components/Pipeline";
import type { CaseStudy, QueryResponse } from "@/lib/api";
import { useTypewriter } from "@/lib/useTypewriter";

export interface Turn {
  id: number;
  question: string;
  strategy: string;
  generate: boolean;
  sourceCount: number;
  caseStudy?: CaseStudy;
  result?: QueryResponse;
  error?: string;
}

interface Reply {
  kind: "answer" | "refusal" | "error" | "info";
  lead?: string;
  segments: { text: string; sources: number[] }[];
}

/** What the assistant says, as plain text segments, so it can be typed out. */
function replyFor(turn: Turn): Reply | null {
  if (turn.error) return { kind: "error", lead: "Something went wrong.", segments: [{ text: turn.error, sources: [] }] };
  const r = turn.result;
  if (!r) return null;
  if (r.status === "answered") return { kind: "answer", segments: r.claims.map((c) => ({ text: c.text, sources: c.sources })) };
  if (r.status === "refused_by_gate" && r.gate) {
    return {
      kind: "refusal",
      lead: "That isn't in your sources.",
      segments: [
        {
          text: `I didn't ask the LLM: the most relevant passage scored ${r.gate.top_score?.toFixed(2)}, below the confidence gate of ${r.gate.threshold.toFixed(1)}, so there was nothing reliable to answer from.`,
          sources: [],
        },
      ],
    };
  }
  if (r.status === "refused_by_model") {
    return { kind: "refusal", lead: "Your sources don't answer this.", segments: [{ text: r.abstain_reason ?? "", sources: [] }] };
  }
  if (r.status === "error") {
    return { kind: "error", lead: "The LLM is unavailable right now.", segments: [{ text: r.error ?? "", sources: [] }] };
  }
  return {
    kind: "info",
    segments: [{ text: "Answer writing is switched off, so here is only what retrieval found. Open the evidence below.", sources: [] }],
  };
}

function Avatar() {
  return (
    <div className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-ink text-[13px] font-semibold text-bg" aria-hidden>
      G
    </div>
  );
}

function Thinking({ turn }: { turn: Turn }) {
  return (
    <p className="shimmer pt-0.5 text-[15px]">
      Searching {turn.sourceCount} source{turn.sourceCount === 1 ? "" : "s"}
      {turn.generate ? " and writing an answer…" : "…"}
    </p>
  );
}

export default function ChatTurn({ turn, animate, onGrow }: { turn: Turn; animate: boolean; onGrow: () => void }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [cited, setCited] = useState<number | null>(null);
  const reply = replyFor(turn);
  const total = reply ? reply.segments.reduce((n, s) => n + s.text.length, 0) : 0;
  const { shown, done } = useTypewriter(total, animate && reply !== null);
  const running = !turn.result && !turn.error;
  const stages = stagesFor(turn.strategy, turn.generate, turn.result ?? null, running);
  const citation = turn.result?.sources.find((s) => s.number === cited);

  // Keep the newest text in view while it types.
  useEffect(() => {
    onGrow();
  }, [shown, done, onGrow]);

  // Where each segment starts in the typed stream, so each shows its own slice.
  const starts = (reply?.segments ?? []).map((_, i, all) =>
    all.slice(0, i).reduce((n, seg) => n + seg.text.length, 0),
  );
  const Icon = reply?.kind === "error" ? TriangleAlert : CircleSlash;

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-end">
        {turn.caseStudy && (
          <span className="mb-1 mr-1 text-[11px] font-medium tracking-wider text-faint uppercase">
            Case study · {turn.caseStudy.label}
          </span>
        )}
        <div className="max-w-[85%] rounded-3xl bg-surface px-4 py-2.5 leading-relaxed whitespace-pre-wrap">
          {turn.question}
        </div>
      </div>

      <div className="flex gap-4">
        <Avatar />
        <div className="min-w-0 flex-1">
          {running && <Thinking turn={turn} />}

          {reply && (
            <div className="text-[15.5px] leading-7">
              {(reply.kind === "refusal" || reply.kind === "error") && (
                <p className="mb-1 flex items-center gap-2 font-semibold">
                  <Icon size={15} />
                  {reply.lead}
                </p>
              )}
              <div className={reply.kind === "answer" ? "space-y-3" : "text-muted"}>
                {reply.segments.map((seg, i) => {
                  const visible = Math.min(seg.text.length, Math.max(0, shown - starts[i]));
                  if (visible === 0 && i > 0) return null;
                  const complete = visible === seg.text.length;
                  const typing = !done && !complete;
                  return (
                    <p key={i} className={typing ? "caret" : undefined}>
                      {seg.text.slice(0, visible)}
                      {complete &&
                        seg.sources.map((n) => (
                          <sup key={n}>
                            <button
                              type="button"
                              onClick={() => setCited(cited === n ? null : n)}
                              className={`fade-up ml-0.5 inline-block min-w-4 rounded-full px-1 text-[10px] leading-4 font-semibold transition-colors ${
                                cited === n ? "bg-ink text-bg" : "bg-surface text-ink hover:bg-line"
                              }`}
                              aria-label={`Citation ${n}`}
                            >
                              {n}
                            </button>
                          </sup>
                        ))}
                    </p>
                  );
                })}
              </div>
            </div>
          )}

          {citation && (
            <figure className="fade-up mt-4 rounded-2xl border border-line px-4 py-3">
              <figcaption className="text-xs text-muted">
                <span className="font-semibold text-ink">[{citation.number}]</span> {citation.title} · page{" "}
                {citation.page_number}
              </figcaption>
              <blockquote className="mt-1.5 text-sm leading-relaxed">{citation.quote}</blockquote>
            </figure>
          )}

          {turn.result && done && (
            <div className="fade-up mt-4">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <Pipeline stages={stages} />
                <button
                  type="button"
                  onClick={() => setShowEvidence(!showEvidence)}
                  className="flex items-center gap-1 text-xs text-muted hover:text-ink"
                  aria-expanded={showEvidence}
                >
                  <ChevronRight size={14} className={`transition-transform ${showEvidence ? "rotate-90" : ""}`} />
                  Evidence · {turn.result.seconds.toFixed(1)}s
                </button>
              </div>
              {turn.caseStudy && <p className="mt-1.5 text-xs text-faint">Expected · {turn.caseStudy.expect}</p>}
              {showEvidence && (
                <div className="fade-up mt-3">
                  <Evidence result={turn.result} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
