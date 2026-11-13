"use client";

import { Check, FileText, Loader2, Plus, X } from "lucide-react";

import type { Source } from "@/lib/api";

interface Props {
  sources: Source[];
  selected: Set<string>;
  uploading: string | null;
  uploadError: string | null;
  onToggle: (filename: string) => void;
  onToggleAll: () => void;
  onPick: () => void;
  onRemove: (filename: string) => void;
}

export default function SourcesPanel(props: Props) {
  const { sources, selected, uploading, uploadError } = props;
  const allSelected = sources.length > 0 && sources.every((s) => selected.has(s.filename));

  return (
    <div className="space-y-4">
      <button
        type="button"
        onClick={props.onPick}
        disabled={uploading !== null}
        className="flex w-full items-center justify-center gap-2 rounded-xl border border-line py-2.5 text-sm font-medium hover:bg-surface disabled:opacity-60"
      >
        {uploading ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />}
        {uploading ? `Adding ${uploading}` : "Add source"}
      </button>
      {uploadError && <p className="text-xs text-bad">{uploadError}</p>}

      {sources.length > 0 && (
        <div>
          <button
            type="button"
            onClick={props.onToggleAll}
            className="mb-1 px-2 text-xs text-muted hover:text-ink"
          >
            {allSelected ? "Deselect all" : "Select all"}
          </button>
          <ul>
            {sources.map((s) => {
              const on = selected.has(s.filename);
              return (
                <li key={s.filename} className="group flex items-center gap-2 rounded-lg px-2 py-2 hover:bg-surface">
                  <FileText size={16} className="shrink-0 text-faint" />
                  <button
                    type="button"
                    onClick={() => props.onToggle(s.filename)}
                    className="min-w-0 flex-1 text-left"
                    aria-pressed={on}
                  >
                    <div className={`truncate text-sm ${on ? "text-ink" : "text-faint"}`} title={s.filename}>
                      {s.filename}
                    </div>
                    <div className="text-xs text-faint">
                      {s.chunks} passages
                      {s.guide_status === "pending" && " · reading…"}
                      {s.guide_status === "failed" && " · no guide"}
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => props.onRemove(s.filename)}
                    className="rounded p-0.5 text-faint opacity-0 group-hover:opacity-100 hover:text-bad focus:opacity-100"
                    aria-label={`Remove ${s.filename}`}
                  >
                    <X size={14} />
                  </button>
                  <button
                    type="button"
                    onClick={() => props.onToggle(s.filename)}
                    className={`grid h-4 w-4 shrink-0 place-items-center rounded border transition-colors ${
                      on ? "border-accent bg-accent text-white" : "border-line"
                    }`}
                    aria-label={`${on ? "Exclude" : "Include"} ${s.filename}`}
                  >
                    {on && <Check size={11} strokeWidth={3} />}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
