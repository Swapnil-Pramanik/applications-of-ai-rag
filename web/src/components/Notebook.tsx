"use client";

import { ArrowUp, BookOpen, Files, RotateCcw, Upload, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import ChatTurn, { type Turn } from "@/components/ChatTurn";
import GuidePanel from "@/components/GuidePanel";
import SourcesPanel from "@/components/SourcesPanel";
import {
  addSource,
  ask,
  getMeta,
  getSources,
  removeSource,
  type CaseStudy,
  type Meta,
  type Source,
} from "@/lib/api";

const ACCEPT = ".pdf,.docx,.txt,.md";

function Drawer({
  side,
  title,
  open,
  onClose,
  children,
}: {
  side: "left" | "right";
  title: string;
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const hidden = side === "left" ? "-translate-x-full" : "translate-x-full";
  return (
    <>
      {/* Backdrop on small screens only; on desktop the chat stays usable beside the drawer. */}
      <div
        onClick={onClose}
        className={`fixed inset-0 top-14 z-20 bg-black/10 transition-opacity lg:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
      />
      <aside
        aria-label={title}
        aria-hidden={!open}
        inert={!open}
        className={`fixed top-14 bottom-0 z-30 w-80 max-w-[85vw] overflow-y-auto bg-bg px-5 py-5 transition-transform duration-300 ease-out ${
          side === "left" ? "left-0 border-r" : "right-0 border-l"
        } border-line ${open ? "translate-x-0" : hidden}`}
      >
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-sm font-semibold">{title}</h2>
          <button type="button" onClick={onClose} className="rounded p-1 text-faint hover:text-ink" aria-label={`Close ${title}`}>
            <X size={16} />
          </button>
        </div>
        {children}
      </aside>
    </>
  );
}

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
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const nextId = useRef(1);
  const bottom = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

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
  // Case studies come from the most recently added selected source.
  const focus = [...selectedSources].reverse().find((s) => s.guide_status !== "failed") ?? selectedSources.at(-1);
  const canAsk = selectedSources.length > 0 && !busy;

  async function onAdd(file: File) {
    setUploading(file.name);
    setUploadError(null);
    try {
      const added = await addSource(file);
      setSources((prev) => [...prev, added]);
      setSelected((prev) => new Set(prev).add(added.filename));
    } catch (e) {
      setUploadError((e as Error).message);
      setSourcesOpen(true);
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
    if (!q || !canAsk) return;
    const id = nextId.current++;
    const names = selectedSources.map((s) => s.filename);
    setQuestion("");
    setTurns((prev) => [...prev, { id, question: q, strategy, generate, sourceCount: names.length, caseStudy }]);
    const update = (patch: Partial<Turn>) =>
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
    try {
      update({ result: await ask(q, strategy, generate, names) });
    } catch (e) {
      update({ error: (e as Error).message });
    }
  }

  const caseButtons = (variant: "list" | "row") =>
    focus?.case_studies.map((c) =>
      variant === "list" ? (
        <button
          key={c.kind}
          type="button"
          disabled={!canAsk}
          onClick={() => run(c.question, c)}
          className="group flex w-full gap-4 border-t border-line py-3 text-left first:border-t-0 disabled:opacity-40"
        >
          <span className="w-28 shrink-0 text-xs font-medium tracking-wide text-faint uppercase">{c.label}</span>
          <span className="text-sm text-muted group-hover:text-ink">{c.question}</span>
        </button>
      ) : (
        <button
          key={c.kind}
          type="button"
          disabled={!canAsk}
          title={`${c.question}\nExpected: ${c.expect}`}
          onClick={() => run(c.question, c)}
          className="text-xs text-muted hover:text-accent disabled:opacity-40"
        >
          {c.label}
        </button>
      ),
    );

  return (
    <div
      className="flex h-full flex-col"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onAdd(file);
      }}
    >
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onAdd(file);
          e.target.value = "";
        }}
      />

      <header className="relative z-40 flex h-14 shrink-0 items-center justify-between border-b border-line bg-bg px-3 sm:px-5">
        <button
          type="button"
          onClick={() => setSourcesOpen(!sourcesOpen)}
          className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm hover:bg-surface ${sourcesOpen ? "bg-surface" : ""}`}
          aria-expanded={sourcesOpen}
        >
          <Files size={16} className="text-muted" />
          Sources
          <span className="text-faint tabular-nums">{sources.length}</span>
        </button>
        <span className="text-sm font-semibold tracking-tight">Grounded Notebook</span>
        <div className="flex items-center gap-1">
          {turns.length > 0 && (
            <button
              type="button"
              onClick={() => setTurns([])}
              className="rounded-lg p-2 text-muted hover:bg-surface hover:text-ink"
              aria-label="Clear chat"
              title="Clear chat"
            >
              <RotateCcw size={15} />
            </button>
          )}
          <button
            type="button"
            onClick={() => setGuideOpen(!guideOpen)}
            className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm hover:bg-surface ${guideOpen ? "bg-surface" : ""}`}
            aria-expanded={guideOpen}
          >
            <BookOpen size={16} className="text-muted" />
            <span className="hidden sm:inline">Guide</span>
          </button>
        </div>
      </header>

      <Drawer side="left" title="Sources" open={sourcesOpen} onClose={() => setSourcesOpen(false)}>
        <SourcesPanel
          sources={sources}
          selected={selected}
          uploading={uploading}
          uploadError={uploadError}
          onToggle={toggle}
          onToggleAll={() =>
            setSelected(
              sources.every((s) => selected.has(s.filename)) ? new Set() : new Set(sources.map((s) => s.filename)),
            )
          }
          onPick={() => fileInput.current?.click()}
          onRemove={onRemove}
        />
      </Drawer>
      <Drawer side="right" title="Guide" open={guideOpen} onClose={() => setGuideOpen(false)}>
        <GuidePanel
          meta={meta}
          guides={selectedSources}
          strategy={strategy}
          generate={generate}
          onStrategy={setStrategy}
          onGenerate={setGenerate}
        />
      </Drawer>

      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl px-5 pt-12 pb-8 sm:pt-16">
          {loadError && <p className="mb-8 text-sm text-bad">{loadError}</p>}

          {sources.length === 0 ? (
            <div className="fade-up pt-8 text-center sm:pt-16">
              <h1 className="text-3xl font-semibold tracking-tight">Ask your documents</h1>
              <p className="mx-auto mt-3 max-w-md text-muted">
                Answers come only from the files you add, every claim is cited, and it tells you when the answer
                isn&apos;t there.
              </p>
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                disabled={uploading !== null}
                className={`mx-auto mt-10 flex w-full max-w-md flex-col items-center gap-2 rounded-2xl border border-dashed px-6 py-12 transition-colors disabled:opacity-60 ${
                  dragging ? "border-accent bg-accent-soft" : "border-line hover:bg-surface"
                }`}
              >
                <Upload size={20} className={uploading ? "animate-pulse text-accent" : "text-muted"} />
                <span className="font-medium">{uploading ? `Adding ${uploading}…` : "Add a source"}</span>
                <span className="text-xs text-faint">PDF, DOCX, TXT or MD · drop it anywhere</span>
              </button>
              {uploadError && <p className="mt-3 text-sm text-bad">{uploadError}</p>}
            </div>
          ) : turns.length === 0 ? (
            <div className="fade-up">
              <p className="text-sm text-faint">
                {selectedSources.length === 0
                  ? "No sources selected"
                  : selectedSources.map((s) => s.filename).join(", ")}
              </p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight">
                {selectedSources.length === 0 ? "Select a source to begin" : "What would you like to know?"}
              </h1>
              {focus?.summary && <p className="mt-4 leading-7 text-muted">{focus.summary}</p>}
              {focus?.guide_status === "pending" && (
                <p className="mt-4 animate-pulse text-sm text-muted">Reading the document to suggest questions…</p>
              )}
              {focus && focus.guide_status !== "pending" && (
                <div className="mt-10">
                  <h2 className="mb-1 text-[11px] font-medium tracking-wider text-faint uppercase">Try a case study</h2>
                  {caseButtons("list")}
                </div>
              )}
              {selectedSources.length === 0 && (
                <button type="button" onClick={() => setSourcesOpen(true)} className="mt-6 text-sm text-accent">
                  Open sources →
                </button>
              )}
            </div>
          ) : (
            <div className="space-y-14">
              {turns.map((turn) => (
                <ChatTurn key={turn.id} turn={turn} />
              ))}
            </div>
          )}
          <div ref={bottom} />
        </div>
      </main>

      <footer className="relative shrink-0 bg-bg px-5 pt-2 pb-4">
        {/* Soft fade so answers scroll away under the composer instead of being cut. */}
        <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-t from-bg to-transparent" />
        <div className="mx-auto max-w-2xl">
          {turns.length > 0 && focus && (
            <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 px-1">
              <span className="text-xs text-faint">Try</span>
              {caseButtons("row")}
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              run(question);
            }}
            className="flex items-center gap-2 rounded-2xl border border-line bg-bg py-1.5 pr-1.5 pl-4 shadow-[0_1px_2px_rgba(0,0,0,0.04)] transition-colors focus-within:border-faint"
          >
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={selectedSources.length === 0}
              placeholder={
                sources.length === 0
                  ? "Add a source to start"
                  : selectedSources.length === 0
                    ? "Select a source first"
                    : "Ask a question"
              }
              className="min-w-0 flex-1 bg-transparent py-1.5 outline-none placeholder:text-faint"
              aria-label="Your question"
            />
            <button
              type="submit"
              disabled={!canAsk || !question.trim()}
              className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-ink text-bg transition-opacity disabled:opacity-20"
              aria-label="Ask"
            >
              <ArrowUp size={16} />
            </button>
          </form>
          <p className="mt-2 text-center text-[11px] text-faint">
            {selectedSources.length > 0
              ? `Answering from ${selectedSources.length} source${selectedSources.length === 1 ? "" : "s"} · check the citations`
              : " "}
          </p>
        </div>
      </footer>
    </div>
  );
}
