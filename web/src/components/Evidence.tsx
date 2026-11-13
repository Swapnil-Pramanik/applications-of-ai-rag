"use client";

import { useState } from "react";

import type { QueryResponse } from "@/lib/api";

/** The retrieved passages; for +rerank retrievers, each score drawn against the gate. */
export default function Evidence({ result }: { result: QueryResponse }) {
  const [open, setOpen] = useState<number | null>(null);
  const { gate, passages } = result;
  if (passages.length === 0) return <p className="text-sm text-muted">No passages were retrieved.</p>;

  // Cross-encoder logits sit roughly in [-11, 11]; widen if a score falls outside.
  const scores = passages.map((p) => p.score);
  const lo = Math.min(-11, ...scores);
  const hi = Math.max(11, ...scores);
  const pos = (x: number) => `${((x - lo) / (hi - lo)) * 100}%`;

  return (
    <div className="space-y-1">
      <p className="pb-1 text-xs text-faint">
        {gate
          ? `Relevance scores from the cross-encoder. The tick marks the gate (${gate.threshold.toFixed(1)}); passages below it never reach the LLM.`
          : `${result.strategy_label}: no cross-encoder scores, so the gate is off.`}
      </p>
      {passages.map((p) => (
        <div key={p.rank}>
          <button
            type="button"
            onClick={() => setOpen(open === p.rank ? null : p.rank)}
            className="group grid w-full grid-cols-[1.25rem_1fr_auto] items-center gap-x-3 rounded-lg px-2 py-1.5 text-left text-sm hover:bg-surface"
            aria-expanded={open === p.rank}
          >
            <span className="text-xs text-faint tabular-nums">{p.rank}</span>
            <span className="min-w-0 truncate text-muted group-hover:text-ink">
              {p.title} <span className="text-faint">· p. {p.page_number}</span>
            </span>
            <span
              className={`text-xs tabular-nums ${p.passed_gate === null ? "text-muted" : p.passed_gate ? "text-good" : "text-bad"}`}
            >
              {p.score.toFixed(2)}
            </span>
            {gate && (
              <span className="relative col-start-2 col-end-4 mt-1.5 h-0.5 rounded-full bg-line">
                <span
                  className={`absolute inset-y-0 left-0 rounded-full ${p.passed_gate ? "bg-good/70" : "bg-bad/60"}`}
                  style={{ width: pos(p.score) }}
                />
                <span className="absolute -top-1 h-2.5 w-px bg-ink/60" style={{ left: pos(gate.threshold) }} />
              </span>
            )}
          </button>
          {open === p.rank && (
            <p className="fade-up mx-2 mb-2 border-l border-line pl-3 text-sm leading-relaxed text-muted">
              {p.text}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
