"use client";

import type { Citation, Meta, Source } from "@/lib/api";

interface Props {
  meta: Meta | null;
  guides: Source[];
  strategy: string;
  generate: boolean;
  citation: Citation | null;
  onStrategy: (key: string) => void;
  onGenerate: (on: boolean) => void;
  onCloseCitation: () => void;
}

export default function StudioPanel(props: Props) {
  const { meta, guides, strategy, generate, citation } = props;
  const threshold = meta?.gate_threshold ?? -2;

  return (
    <div className="space-y-6">
      {citation && (
        <section className="rise rounded-xl border border-accent/40 bg-accent-soft p-4">
          <div className="flex items-start justify-between gap-2">
            <h2 className="text-sm font-semibold text-accent">Citation [{citation.number}]</h2>
            <button type="button" onClick={props.onCloseCitation} className="text-muted hover:text-ink" aria-label="Close citation">
              ×
            </button>
          </div>
          <p className="mt-1 text-sm font-medium">
            {citation.title} <span className="font-normal text-muted">· page {citation.page_number}</span>
          </p>
          <blockquote className="mt-2 border-l-2 border-accent/50 pl-3 font-serif text-[15px] leading-relaxed">
            {citation.quote}
          </blockquote>
        </section>
      )}

      <section>
        <h2 className="font-semibold">Notebook guide</h2>
        {guides.length === 0 ? (
          <p className="mt-2 text-sm text-muted">Select a source to see its summary.</p>
        ) : (
          <ul className="mt-2 space-y-3">
            {guides.map((s) => (
              <li key={s.filename}>
                <div className="truncate text-xs font-semibold text-muted">{s.filename}</div>
                <p className="mt-0.5 text-sm leading-relaxed">
                  {s.summary ??
                    (s.guide_status === "pending" ? "Writing a summary…" : "No summary (the LLM was unavailable).")}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="font-semibold">Settings</h2>
        <p className="mt-2 text-sm">
          <span className="text-muted">LLM</span>{" "}
          <span className="font-medium">{meta?.llm ?? "…"}</span>
        </p>
        <label className="mt-2 block text-sm">
          <span className="text-muted">Retriever</span>
          <select
            value={strategy}
            onChange={(e) => props.onStrategy(e.target.value)}
            className="mt-1 w-full rounded-lg border border-line bg-card px-2 py-1.5"
          >
            {(meta?.strategies ?? [{ key: "hybrid+rerank", label: "Hybrid + rerank (default)" }]).map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <div className="mt-3 flex items-center gap-3 text-sm">
          <button
            type="button"
            role="switch"
            aria-checked={generate}
            aria-label="Generate answers with the LLM"
            onClick={() => props.onGenerate(!generate)}
            className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${generate ? "bg-accent" : "bg-line"}`}
          >
            <span
              className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${generate ? "translate-x-5" : ""}`}
            />
          </button>
          <span>
            Answer with the LLM
            <span className="block text-xs text-muted">
              {generate ? "Turn off to show retrieval only (no API quota)" : "Off: retrieval only, no API quota"}
            </span>
          </span>
        </div>
      </section>

      <section>
        <h2 className="font-semibold">How it works</h2>
        <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-muted">
          <li><b className="text-ink">Retrieve</b>: dense (BGE + FAISS) and BM25 keyword search each find 20 passages in the selected sources.</li>
          <li><b className="text-ink">Fuse</b>: Reciprocal Rank Fusion merges the two lists.</li>
          <li><b className="text-ink">Rerank</b>: a cross-encoder rescores them and keeps the top 5.</li>
          <li><b className="text-ink">Gate</b>: if no passage scores above {threshold.toFixed(1)}, refuse without calling the LLM.</li>
          <li><b className="text-ink">Answer</b>: the LLM ({meta?.llm ?? "loading…"}) answers only from those passages and cites each claim; invented or missing citations make it refuse.</li>
        </ol>
        <p className="mt-2 text-xs text-muted">
          Nothing is trained or fine-tuned: all three models are used as-is.
        </p>
      </section>
    </div>
  );
}
