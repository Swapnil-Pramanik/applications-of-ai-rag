"use client";

import type { QueryResponse } from "@/lib/api";

interface Props {
  result: QueryResponse;
  activeSource: number | null;
  onSource: (n: number | null) => void;
}

function Banner({ tone, title, children }: { tone: "warn" | "bad" | "info"; title: string; children: React.ReactNode }) {
  const style = {
    warn: "border-warn/40 bg-warn-soft text-warn",
    bad: "border-bad/40 bg-bad-soft text-bad",
    info: "border-accent/30 bg-accent-soft text-accent",
  }[tone];
  return (
    <div className={`rounded-xl border p-4 ${style}`}>
      <div className="font-semibold">{title}</div>
      <div className="mt-1 text-sm text-ink">{children}</div>
    </div>
  );
}

export default function AnswerPanel({ result, activeSource, onSource }: Props) {
  const { status, gate } = result;

  return (
    <section className="rise rounded-2xl border border-line bg-card p-6 shadow-sm">
      <h2 className="font-serif text-2xl font-semibold">Answer</h2>

      <div className="mt-4 space-y-4">
        {status === "refused_by_gate" && gate && (
          <Banner tone="warn" title="Refused before calling the LLM">
            No retrieved passage looked relevant enough. The best cross-encoder score was{" "}
            <strong>{gate.top_score?.toFixed(2)}</strong>; the gate needs at least{" "}
            <strong>{gate.threshold.toFixed(1)}</strong>. Not calling Gemini here is what keeps the
            system from guessing.
          </Banner>
        )}
        {status === "refused_by_model" && (
          <Banner tone="warn" title="The model read the passages and judged them insufficient">
            {result.abstain_reason}
          </Banner>
        )}
        {status === "retrieval_only" && (
          <Banner tone="info" title="Generation is off">
            Showing only what the retriever found. This uses no Gemini quota.
          </Banner>
        )}
        {status === "error" && (
          <Banner tone="bad" title="Gemini is unavailable">
            {result.error} An outage is reported as an error, never counted as a refusal.
          </Banner>
        )}

        {status === "answered" && (
          <>
            <div className="space-y-3 font-serif text-xl leading-relaxed">
              {result.claims.map((claim, i) => (
                <p key={i}>
                  {claim.text}
                  {claim.sources.map((n) => (
                    <button
                      key={n}
                      type="button"
                      onMouseEnter={() => onSource(n)}
                      onMouseLeave={() => onSource(null)}
                      onFocus={() => onSource(n)}
                      onBlur={() => onSource(null)}
                      className={`ml-1 inline-flex h-6 min-w-6 -translate-y-1 items-center justify-center rounded-md px-1.5 align-middle font-sans text-xs font-semibold transition-colors ${
                        activeSource === n ? "bg-accent text-white" : "bg-accent-soft text-accent"
                      }`}
                      aria-label={`Source ${n}`}
                    >
                      {n}
                    </button>
                  ))}
                </p>
              ))}
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted">Sources</h3>
              <ul className="mt-2 space-y-2">
                {result.sources.map((source) => (
                  <li
                    key={source.number}
                    onMouseEnter={() => onSource(source.number)}
                    onMouseLeave={() => onSource(null)}
                    className={`rounded-xl border p-3 transition-colors ${
                      activeSource === source.number ? "border-accent bg-accent-soft" : "border-line"
                    }`}
                  >
                    <div className="flex items-baseline gap-2 text-sm">
                      <span className="font-semibold text-accent">[{source.number}]</span>
                      <span className="font-semibold">{source.title}</span>
                      <span className="text-muted">page {source.page_number}</span>
                    </div>
                    <blockquote className="mt-2 border-l-2 border-accent/40 pl-3 text-sm text-muted italic">
                      {source.quote}
                    </blockquote>
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
