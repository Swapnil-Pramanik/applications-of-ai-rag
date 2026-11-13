"use client";

import { useState } from "react";

import type { QueryResponse } from "@/lib/api";

/** Top-k passages with their scores; for +rerank arms, a bar against the gate threshold. */
export default function EvidencePanel({ result }: { result: QueryResponse }) {
  const [open, setOpen] = useState<number | null>(null);
  const { gate, passages } = result;

  // Cross-encoder logits are roughly in [-11, 11]; widen if a score falls outside.
  const scores = passages.map((p) => p.score);
  const lo = Math.min(-11, ...scores);
  const hi = Math.max(11, ...scores);
  const pos = (x: number) => `${((x - lo) / (hi - lo)) * 100}%`;

  return (
    <section className="rise rounded-2xl border border-line bg-card p-6 shadow-sm">
      <h2 className="font-serif text-2xl font-semibold">Evidence</h2>
      <p className="mt-1 text-sm text-muted">
        {gate
          ? `Top ${passages.length} passages by cross-encoder score. The line marks the gate (${gate.threshold.toFixed(1)}); passages below it are never shown to the LLM.`
          : `Top ${passages.length} passages. ${result.strategy_label} scores are not cross-encoder scores, so the gate is off.`}
      </p>

      <ol className="mt-4 space-y-2">
        {passages.map((p) => (
          <li key={p.rank} className="rounded-xl border border-line">
            <button
              type="button"
              onClick={() => setOpen(open === p.rank ? null : p.rank)}
              className="w-full p-3 text-left"
              aria-expanded={open === p.rank}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold">
                    <span className="text-muted">#{p.rank}</span> {p.title}
                  </div>
                  <div className="text-xs text-muted">page {p.page_number}</div>
                </div>
                <div className="shrink-0 text-right">
                  <div className="font-mono text-sm">{p.score.toFixed(3)}</div>
                  {p.passed_gate !== null && (
                    <div className={`text-xs font-medium ${p.passed_gate ? "text-good" : "text-bad"}`}>
                      {p.passed_gate ? "passed gate" : "below gate"}
                    </div>
                  )}
                </div>
              </div>
              {gate && (
                <div className="relative mt-2 h-1.5 rounded-full bg-line">
                  <div
                    className={`absolute inset-y-0 left-0 rounded-full ${p.passed_gate ? "bg-good" : "bg-bad"}`}
                    style={{ width: pos(p.score) }}
                  />
                  <div className="absolute -inset-y-1 w-0.5 bg-ink" style={{ left: pos(gate.threshold) }} />
                </div>
              )}
            </button>
            {open === p.rank && (
              <p className="border-t border-line p-3 text-sm leading-relaxed text-muted">{p.text}</p>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
