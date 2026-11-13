"use client";

import { useState } from "react";

import Evidence from "@/components/Evidence";
import Pipeline, { stagesFor } from "@/components/Pipeline";
import type { CaseStudy, Citation, QueryResponse } from "@/lib/api";

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

function Banner({ tone, title, children }: { tone: "warn" | "bad" | "info"; title: string; children: React.ReactNode }) {
  const style = {
    warn: "border-warn/40 bg-warn-soft",
    bad: "border-bad/40 bg-bad-soft",
    info: "border-accent/30 bg-accent-soft",
  }[tone];
  const titleColor = { warn: "text-warn", bad: "text-bad", info: "text-accent" }[tone];
  return (
    <div className={`rounded-xl border p-3 ${style}`}>
      <div className={`text-sm font-semibold ${titleColor}`}>{title}</div>
      <div className="mt-1 text-sm">{children}</div>
    </div>
  );
}

function Body({ result, onCite }: { result: QueryResponse; onCite: (c: Citation) => void }) {
  const { status, gate } = result;
  if (status === "refused_by_gate" && gate) {
    return (
      <Banner tone="warn" title="Refused before calling the LLM">
        Nothing in the selected sources looked relevant: the best passage scored{" "}
        <strong>{gate.top_score?.toFixed(2)}</strong>, below the gate of{" "}
        <strong>{gate.threshold.toFixed(1)}</strong>. Gemini was never asked, so it could not guess.
      </Banner>
    );
  }
  if (status === "refused_by_model") {
    return (
      <Banner tone="warn" title="The sources don't answer this">
        Gemini read the top passages and judged them insufficient: {result.abstain_reason}
      </Banner>
    );
  }
  if (status === "error") {
    return (
      <Banner tone="bad" title="Gemini is unavailable">
        {result.error} An outage is an error, never counted as a refusal.
      </Banner>
    );
  }
  if (status === "retrieval_only") {
    return (
      <Banner tone="info" title="Generation is off">
        Showing only what retrieval found (no Gemini quota used). Open the evidence below.
      </Banner>
    );
  }

  const byNumber = new Map(result.sources.map((s) => [s.number, s]));
  return (
    <div className="space-y-2 font-serif text-lg leading-relaxed">
      {result.claims.map((claim, i) => (
        <p key={i}>
          {claim.text}
          {claim.sources.map((n) => {
            const cite = byNumber.get(n);
            return (
              <button
                key={n}
                type="button"
                onClick={() => cite && onCite(cite)}
                title={cite ? `${cite.title}, page ${cite.page_number}` : undefined}
                className="ml-1 inline-flex h-5 min-w-5 -translate-y-0.5 items-center justify-center rounded-full bg-accent-soft px-1.5 align-middle font-sans text-[11px] font-semibold text-accent hover:bg-accent hover:text-white"
                aria-label={`Citation ${n}`}
              >
                {n}
              </button>
            );
          })}
        </p>
      ))}
    </div>
  );
}

export default function ChatTurn({ turn, onCite }: { turn: Turn; onCite: (c: Citation) => void }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const running = !turn.result && !turn.error;
  const stages = stagesFor(turn.strategy, turn.generate, turn.result ?? null, running);

  return (
    <div className="rise space-y-3">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-white">
          {turn.caseStudy && (
            <div className="mb-1 text-xs font-semibold tracking-wide text-white/80 uppercase">
              Case study · {turn.caseStudy.label}
            </div>
          )}
          {turn.question}
        </div>
      </div>

      <div className="rounded-2xl rounded-tl-md border border-line bg-card p-4 shadow-sm">
        <Pipeline stages={stages} />
        {turn.caseStudy && (
          <p className="mt-2 text-xs text-muted">
            <span className="font-semibold">Expected:</span> {turn.caseStudy.expect}
          </p>
        )}

        <div className="mt-3">
          {running && (
            <p className="animate-pulse text-sm text-muted">
              Searching {turn.sourceCount} source{turn.sourceCount === 1 ? "" : "s"}
              {turn.generate ? " and asking Gemini…" : "…"}
            </p>
          )}
          {turn.error && <Banner tone="bad" title="Request failed">{turn.error}</Banner>}
          {turn.result && <Body result={turn.result} onCite={onCite} />}
        </div>

        {turn.result && (
          <div className="mt-3 border-t border-line pt-2">
            <button
              type="button"
              onClick={() => setShowEvidence(!showEvidence)}
              className="text-xs font-medium text-muted hover:text-ink"
              aria-expanded={showEvidence}
            >
              {showEvidence ? "▾" : "▸"} Evidence: {turn.result.passages.length} passages ·{" "}
              {turn.result.strategy_label} · {turn.result.seconds.toFixed(2)}s
            </button>
            {showEvidence && (
              <div className="mt-2">
                <Evidence result={turn.result} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
