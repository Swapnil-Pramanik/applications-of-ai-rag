"use client";

import type { Meta, Source } from "@/lib/api";

interface Props {
  meta: Meta | null;
  guides: Source[];
  strategy: string;
  generate: boolean;
  onStrategy: (key: string) => void;
  onGenerate: (on: boolean) => void;
}

function Heading({ children }: { children: React.ReactNode }) {
  return <h3 className="text-[11px] font-medium tracking-wider text-faint uppercase">{children}</h3>;
}

export default function GuidePanel(props: Props) {
  const { meta, guides, strategy, generate } = props;
  const threshold = meta?.gate_threshold ?? -2;

  return (
    <div className="space-y-8">
      <section>
        <Heading>Summary</Heading>
        {guides.length === 0 ? (
          <p className="mt-2 text-sm text-muted">Select a source to see its summary.</p>
        ) : (
          <div className="mt-2 space-y-4">
            {guides.map((s) => (
              <div key={s.filename}>
                <div className="truncate text-xs text-faint">{s.filename}</div>
                <p className="mt-1 text-sm leading-relaxed">
                  {s.summary ??
                    (s.guide_status === "pending" ? "Reading the document…" : "No summary (the LLM was unavailable).")}
                </p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <Heading>Settings</Heading>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted">Model</span>
          <span>{meta?.llm ?? "…"}</span>
        </div>
        <label className="block text-sm">
          <span className="text-muted">Retriever</span>
          <select
            value={strategy}
            onChange={(e) => props.onStrategy(e.target.value)}
            className="mt-1.5 w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-sm"
          >
            {(meta?.strategies ?? [{ key: "hybrid+rerank", label: "Hybrid + rerank (default)" }]).map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-muted">
            Write answers
            <span className="block text-xs text-faint">Off: retrieval only, no API use</span>
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={generate}
            aria-label="Write answers with the LLM"
            onClick={() => props.onGenerate(!generate)}
            className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${generate ? "bg-accent" : "bg-line"}`}
          >
            <span
              className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-bg shadow-sm transition-transform ${generate ? "translate-x-4" : ""}`}
            />
          </button>
        </div>
      </section>

      <section>
        <Heading>How it works</Heading>
        <ol className="mt-2 space-y-2 text-sm text-muted">
          {[
            ["Retrieve", "Dense (BGE + FAISS) and BM25 keyword search each find 20 passages in your selected sources."],
            ["Fuse", "Reciprocal Rank Fusion merges the two lists."],
            ["Rerank", "A cross-encoder rescores them and keeps the top 5."],
            ["Gate", `If no passage scores above ${threshold.toFixed(1)}, refuse without calling the LLM.`],
            ["Answer", "The LLM answers only from those passages and cites each claim."],
          ].map(([name, text], i) => (
            <li key={name} className="flex gap-3">
              <span className="w-3 shrink-0 text-faint tabular-nums">{i + 1}</span>
              <span>
                <span className="text-ink">{name}.</span> {text}
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-xs text-faint">Nothing is trained or fine-tuned; all models are used as-is.</p>
      </section>
    </div>
  );
}
