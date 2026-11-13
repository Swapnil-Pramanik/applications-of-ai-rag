"use client";

import { useRef, useState } from "react";

import type { Source } from "@/lib/api";

interface Props {
  sources: Source[];
  selected: Set<string>;
  uploading: string | null;
  uploadError: string | null;
  maxMb: number;
  onToggle: (filename: string) => void;
  onToggleAll: () => void;
  onAdd: (file: File) => void;
  onRemove: (filename: string) => void;
}

function GuideBadge({ source }: { source: Source }) {
  if (source.guide_status === "pending")
    return <span className="animate-pulse text-xs text-accent">writing guide…</span>;
  if (source.guide_status === "failed")
    return (
      <span className="text-xs text-warn" title={source.guide_error ?? undefined}>
        no guide (Gemini unavailable)
      </span>
    );
  return null;
}

export default function SourcesPanel(props: Props) {
  const { sources, selected, uploading, uploadError, maxMb } = props;
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const allSelected = sources.length > 0 && sources.every((s) => selected.has(s.filename));

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Sources</h2>
        <span className="text-xs text-muted">{sources.length}</span>
      </div>

      <input
        ref={input}
        type="file"
        accept=".pdf,.docx,.txt,.md"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) props.onAdd(file);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        disabled={uploading !== null}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) props.onAdd(file);
        }}
        className={`mt-3 rounded-xl border border-dashed px-3 text-sm transition-colors disabled:opacity-60 ${
          sources.length === 0 ? "py-8" : "py-3"
        } ${dragging ? "border-accent bg-accent-soft text-accent" : "border-line hover:border-accent hover:text-accent"}`}
      >
        {uploading ? (
          <span className="animate-pulse">Indexing {uploading}…</span>
        ) : (
          <>
            <span className="font-medium">+ Add source</span>
            <span className="block text-xs text-muted">PDF, DOCX, TXT or MD · up to {maxMb} MB · or drop here</span>
          </>
        )}
      </button>
      {uploadError && <p className="mt-2 text-xs text-bad">{uploadError}</p>}

      {sources.length > 0 && (
        <>
          <label className="mt-4 flex cursor-pointer items-center gap-2 px-1 text-xs text-muted">
            <input type="checkbox" checked={allSelected} onChange={props.onToggleAll} className="accent-[var(--accent)]" />
            Select all sources
          </label>
          <ul className="mt-2 space-y-1 overflow-y-auto">
            {sources.map((s) => (
              <li key={s.filename} className="group flex items-start gap-2 rounded-lg px-1 py-1.5 hover:bg-paper">
                <input
                  type="checkbox"
                  checked={selected.has(s.filename)}
                  onChange={() => props.onToggle(s.filename)}
                  className="mt-1 accent-[var(--accent)]"
                  aria-label={`Use ${s.filename}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium" title={s.filename}>
                    {s.filename}
                  </div>
                  <div className="flex gap-2 text-xs text-muted">
                    <span>{s.chunks} passages</span>
                    <GuideBadge source={s} />
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => props.onRemove(s.filename)}
                  className="rounded px-1 text-muted opacity-60 group-hover:opacity-100 hover:text-bad"
                  aria-label={`Remove ${s.filename}`}
                  title="Remove source"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
