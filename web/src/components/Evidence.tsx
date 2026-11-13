"use client";

import { useState } from "react";

import type { QueryResponse } from "@/lib/api";

/** The retrieved passages, with each score drawn against the gate threshold. */
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
    <div>
      <p className="text-xs text-muted">
        {gate
          ? `Cross-encoder scores. The black tick is the gate (${gate.threshold.toFixed(1)}); passages below it are never shown to Gemini.`
          : `${result.strategy_label}: these scores are not cross-encoder scores, so the gate is off.`}
      </p>
      <ol className="mt-2 space-y-1.5">
        {passages.map((p) => (
          <li key={p.rank} className="rounded-lg border border-line">
            <button
              type="button"
              onClick={() => setOpen(open === p.rank ? null : p.rank)}
              className="w-full px-3 py-2 text-left"
              aria-expanded={open === p.rank}
            >
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="min-w-0 truncate">
                  <span className="text-muted">#{p.rank}</span> {p.title}{" "}
                  <span className="text-muted">· p. {p.page_number}</span>
                </span>
                <span className="shrink-0 font-mono text-xs">
                  {p.score.toFixed(2)}
                  {p.passed_gate !== null && (
                    <span className={`ml-2 font-sans font-medium ${p.passed_gate ? "text-good" : "text-bad"}`}>
                      {p.passed_gate ? "passed" : "below gate"}
                    </span>
                  )}
                </span>
              </div>
              {gate && (
                <div className="relative mt-1.5 h-1 rounded-full bg-line">
                  <div
                    className={`absolute inset-y-0 left-0 rounded-full ${p.passed_gate ? "bg-good" : "bg-bad"}`}
                    style={{ width: pos(p.score) }}
                  />
                  <div className="absolute -inset-y-1 w-0.5 bg-ink" style={{ left: pos(gate.threshold) }} />
                </div>
              )}
            </button>
            {open === p.rank && (
              <p className="border-t border-line px-3 py-2 text-sm leading-relaxed text-muted">{p.text}</p>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
