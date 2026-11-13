"use client";

import { ChevronRight, CircleSlash, TriangleAlert } from "lucide-react";
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

function Note({ tone, icon, children }: { tone: "warn" | "bad" | "muted"; icon: React.ReactNode; children: React.ReactNode }) {
  const color = { warn: "text-warn", bad: "text-bad", muted: "text-muted" }[tone];
  return (
    <div className="flex gap-2.5 leading-7">
      <span className={`mt-1.5 shrink-0 ${color}`}>{icon}</span>
      <div className="text-muted">{children}</div>
    </div>
  );
}

function Body({ result, cited, onCite }: { result: QueryResponse; cited: number | null; onCite: (n: number) => void }) {
  const { status, gate } = result;
  if (status === "refused_by_gate" && gate) {
    return (
      <Note tone="warn" icon={<CircleSlash size={16} />}>
        <span className="text-ink">Not in your sources.</span> Refused before calling the LLM: the best
        passage scored <span className="tabular-nums">{gate.top_score?.toFixed(2)}</span>, below the gate
        of <span className="tabular-nums">{gate.threshold.toFixed(1)}</span>.
      </Note>
    );
  }
  if (status === "refused_by_model") {
    return (
      <Note tone="warn" icon={<CircleSlash size={16} />}>
        <span className="text-ink">Your sources don&apos;t answer this.</span> {result.abstain_reason}
      </Note>
    );
  }
  if (status === "error") {
    return (
      <Note tone="bad" icon={<TriangleAlert size={16} />}>
        <span className="text-ink">The LLM is unavailable.</span> {result.error}
      </Note>
    );
  }
  if (status === "retrieval_only") {
    return (
      <Note tone="muted" icon={<ChevronRight size={16} />}>
        Answers are switched off. Open the evidence to see what retrieval found.
      </Note>
    );
  }
  return (
    <div className="space-y-3 text-[16px] leading-7">
      {result.claims.map((claim, i) => (
        <p key={i}>
          {claim.text}
          {claim.sources.map((n) => (
            <sup key={n}>
              <button
                type="button"
                onClick={() => onCite(n)}
                className={`ml-0.5 inline-block rounded px-1 text-[11px] leading-4 font-semibold transition-colors ${
                  cited === n ? "bg-accent text-white" : "text-accent hover:bg-accent-soft"
                }`}
                aria-label={`Citation ${n}`}
              >
                {n}
              </button>
            </sup>
          ))}
        </p>
      ))}
    </div>
  );
}

function CitationQuote({ citation }: { citation: Citation }) {
  return (
    <figure className="fade-up mt-4 rounded-xl bg-surface px-4 py-3">
      <figcaption className="text-xs text-muted">
        <span className="font-semibold text-accent">[{citation.number}]</span> {citation.title} · page{" "}
        {citation.page_number}
      </figcaption>
      <blockquote className="mt-1.5 text-sm leading-relaxed">{citation.quote}</blockquote>
    </figure>
  );
}

export default function ChatTurn({ turn }: { turn: Turn }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [cited, setCited] = useState<number | null>(null);
  const running = !turn.result && !turn.error;
  const stages = stagesFor(turn.strategy, turn.generate, turn.result ?? null, running);
  const citation = turn.result?.sources.find((s) => s.number === cited);

  return (
    <article className="fade-up">
      {turn.caseStudy && (
        <div className="mb-1 text-[11px] font-medium tracking-wider text-faint uppercase">
          Case study · {turn.caseStudy.label}
        </div>
      )}
      <h2 className="text-xl font-semibold tracking-tight">{turn.question}</h2>

      <div className="mt-4">
        {running && (
          <p className="flex items-center gap-2 text-muted">
            <span className="inline-flex gap-1" aria-hidden>
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-faint [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-faint [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-faint" />
            </span>
            Reading {turn.sourceCount} source{turn.sourceCount === 1 ? "" : "s"}
          </p>
        )}
        {turn.error && (
          <Note tone="bad" icon={<TriangleAlert size={16} />}>
            {turn.error}
          </Note>
        )}
        {turn.result && (
          <Body result={turn.result} cited={cited} onCite={(n) => setCited(cited === n ? null : n)} />
        )}
        {citation && <CitationQuote citation={citation} />}
      </div>

      <div className="mt-5 border-t border-line pt-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Pipeline stages={stages} />
          {turn.result && (
            <button
              type="button"
              onClick={() => setShowEvidence(!showEvidence)}
              className="flex items-center gap-1 text-xs text-muted hover:text-ink"
              aria-expanded={showEvidence}
            >
              <ChevronRight size={14} className={`transition-transform ${showEvidence ? "rotate-90" : ""}`} />
              Evidence · {turn.result.seconds.toFixed(1)}s
            </button>
          )}
        </div>
        {turn.caseStudy && (
          <p className="mt-1.5 text-xs text-faint">Expected · {turn.caseStudy.expect}</p>
        )}
        {showEvidence && turn.result && (
          <div className="fade-up mt-3">
            <Evidence result={turn.result} />
          </div>
        )}
      </div>
    </article>
  );
}
