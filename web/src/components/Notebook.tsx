"use client";

import { useEffect, useRef, useState } from "react";

import ChatTurn, { type Turn } from "@/components/ChatTurn";
import SourcesPanel from "@/components/SourcesPanel";
import StudioPanel from "@/components/StudioPanel";
import {
  addSource,
  ask,
  getMeta,
  getSources,
  removeSource,
  type CaseStudy,
  type Citation,
  type Meta,
  type Source,
} from "@/lib/api";

export default function Notebook() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState("hybrid+rerank");
  const [generate, setGenerate] = useState(true);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [citation, setCitation] = useState<Citation | null>(null);
  const nextId = useRef(1);
  const bottom = useRef<HTMLDivElement>(null);

  // Initial load: the API's in-memory notebook may already hold sources.
  useEffect(() => {
    Promise.all([getMeta(), getSources()]).then(
      ([m, s]) => {
        setMeta(m);
        setSources(s);
        setSelected(new Set(s.map((x) => x.filename)));
      },
      (e: Error) => setLoadError(e.message),
    );
  }, []);

  // Guides are written in the background; poll until none is pending.
  const pending = sources.some((s) => s.guide_status === "pending");
  useEffect(() => {
    if (!pending) return;
    const timer = setInterval(() => {
      getSources().then(setSources, () => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, [pending]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const busy = turns.some((t) => !t.result && !t.error);
  const selectedSources = sources.filter((s) => selected.has(s.filename));
  // Case studies come from the most recently added selected source with a guide.
  const focus = [...selectedSources].reverse().find((s) => s.guide_status !== "failed") ?? selectedSources.at(-1);

  async function onAdd(file: File) {
    setUploading(file.name);
    setUploadError(null);
    try {
      const added = await addSource(file);
      setSources((prev) => [...prev, added]);
      setSelected((prev) => new Set(prev).add(added.filename));
    } catch (e) {
      setUploadError((e as Error).message);
    } finally {
      setUploading(null);
    }
  }

  async function onRemove(filename: string) {
    try {
      await removeSource(filename);
      setSources((prev) => prev.filter((s) => s.filename !== filename));
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(filename);
        return next;
      });
    } catch (e) {
      setUploadError((e as Error).message);
    }
  }

  function toggle(filename: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  }

  async function run(text: string, caseStudy?: CaseStudy) {
    const q = text.trim();
    if (!q || busy || selectedSources.length === 0) return;
    const id = nextId.current++;
    const names = selectedSources.map((s) => s.filename);
    setQuestion("");
    setTurns((prev) => [
      ...prev,
      { id, question: q, strategy, generate, sourceCount: names.length, caseStudy },
    ]);
    const update = (patch: Partial<Turn>) =>
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
    try {
      update({ result: await ask(q, strategy, generate, names) });
    } catch (e) {
      update({ error: (e as Error).message });
    }
  }

  const canAsk = selectedSources.length > 0 && !busy;

  return (
    <div className="flex min-h-screen flex-col lg:h-screen">
      <header className="flex items-center justify-between border-b border-line bg-card px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent font-serif text-lg font-bold text-white">G</div>
          <div>
            <div className="leading-tight font-semibold">Grounded Notebook</div>
            <div className="text-xs text-muted">Answers only from your sources, with citations</div>
          </div>
        </div>
        {turns.length > 0 && (
          <button
            type="button"
            onClick={() => {
              setTurns([]);
              setCitation(null);
            }}
            className="rounded-lg border border-line px-3 py-1 text-sm text-muted hover:text-ink"
          >
            Clear chat
          </button>
        )}
      </header>

      {loadError && <div className="border-b border-bad/40 bg-bad-soft px-4 py-2 text-sm text-bad">{loadError}</div>}

      <div className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
        <aside className="border-b border-line bg-card p-4 lg:w-72 lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-b-0">
          <SourcesPanel
            sources={sources}
            selected={selected}
            uploading={uploading}
            uploadError={uploadError}
            maxMb={meta?.max_upload_mb ?? 20}
            onToggle={toggle}
            onToggleAll={() =>
              setSelected(
                sources.every((s) => selected.has(s.filename)) ? new Set() : new Set(sources.map((s) => s.filename)),
              )
            }
            onAdd={onAdd}
            onRemove={onRemove}
          />
        </aside>

        <main className="flex min-w-0 flex-1 flex-col lg:min-h-0">
          <div className="flex-1 px-4 py-6 lg:overflow-y-auto">
            <div className="mx-auto max-w-3xl space-y-6">
              {sources.length === 0 ? (
                <div className="py-16 text-center">
                  <h1 className="font-serif text-3xl font-semibold">Add a source to get started</h1>
                  <p className="mx-auto mt-2 max-w-md text-muted">
                    Upload a PDF, Word file or text note. Questions are answered only from the sources you
                    select, every claim is cited, and the notebook says so when your sources don&apos;t
                    contain the answer.
                  </p>
                </div>
              ) : turns.length === 0 ? (
                <div className="py-10 text-center">
                  <h1 className="font-serif text-3xl font-semibold">
                    {selectedSources.length === 0
                      ? "Select a source to ask about"
                      : `Ask about ${selectedSources.length === 1 ? selectedSources[0].filename : `${selectedSources.length} sources`}`}
                  </h1>
                  {focus?.summary && <p className="mx-auto mt-3 max-w-xl leading-relaxed text-muted">{focus.summary}</p>}
                  <p className="mt-4 text-sm text-muted">Type a question below, or run a case study to see each behaviour.</p>
                </div>
              ) : (
                turns.map((turn) => <ChatTurn key={turn.id} turn={turn} onCite={setCitation} />)
              )}
              <div ref={bottom} />
            </div>
          </div>

          <div className="border-t border-line bg-card px-4 py-3">
            <div className="mx-auto max-w-3xl">
              {focus && (
                <div className="mb-2 flex flex-wrap items-center gap-1.5">
                  <span className="mr-1 text-xs text-muted">
                    Case studies{selectedSources.length > 1 ? ` (${focus.filename})` : ""}:
                  </span>
                  {focus.case_studies.map((c) => (
                    <button
                      key={c.kind}
                      type="button"
                      disabled={!canAsk}
                      title={`${c.question}\nExpected: ${c.expect}`}
                      onClick={() => run(c.question, c)}
                      className="rounded-full border border-line px-2.5 py-0.5 text-xs hover:border-accent hover:text-accent disabled:opacity-40"
                    >
                      {c.label}
                    </button>
                  ))}
                  {focus.guide_status === "pending" && (
                    <span className="animate-pulse text-xs text-accent">writing more case studies…</span>
                  )}
                </div>
              )}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  run(question);
                }}
                className="flex gap-2 rounded-2xl border border-line bg-paper p-1.5 focus-within:border-accent"
              >
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={selectedSources.length === 0}
                  placeholder={
                    sources.length === 0
                      ? "Add a source first"
                      : selectedSources.length === 0
                        ? "Select at least one source"
                        : "Ask a question about your sources…"
                  }
                  className="min-w-0 flex-1 bg-transparent px-3 py-2 outline-none placeholder:text-muted/70"
                  aria-label="Your question"
                />
                <button
                  type="submit"
                  disabled={!canAsk || !question.trim()}
                  className="rounded-xl bg-accent px-5 font-semibold text-white disabled:opacity-40"
                >
                  {busy ? "…" : "Ask"}
                </button>
              </form>
              <p className="mt-1.5 text-center text-[11px] text-muted">
                Searching {selectedSources.length} of {sources.length} source{sources.length === 1 ? "" : "s"}.
                Answers can still be wrong; check the citations.
              </p>
            </div>
          </div>
        </main>

        <aside className="border-t border-line bg-card p-4 lg:w-80 lg:shrink-0 lg:overflow-y-auto lg:border-t-0 lg:border-l">
          <StudioPanel
            meta={meta}
            guides={selectedSources}
            strategy={strategy}
            generate={generate}
            citation={citation}
            onStrategy={setStrategy}
            onGenerate={setGenerate}
            onCloseCitation={() => setCitation(null)}
          />
        </aside>
      </div>
    </div>
  );
}
