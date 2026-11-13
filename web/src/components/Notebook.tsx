"use client";

import { ArrowUp, PanelLeft, PanelRight, SquarePen, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

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
const DESKTOP = "(min-width: 1024px)";

/**
 * A side column. `open === null` means "not toggled yet": open on desktop, closed on
 * phones, decided by CSS so the server render and the first client render agree.
 * On desktop it is a real column that pushes the chat; on phones it slides over it.
 */
function SidePanel({
  side,
  title,
  open,
  onToggle,
  children,
}: {
  side: "left" | "right";
  title: string;
  open: boolean | null;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  const left = side === "left";
  const state =
    open === null
      ? `${left ? "-translate-x-full" : "translate-x-full"} lg:translate-x-0 lg:w-72`
      : open
        ? "translate-x-0 lg:w-72"
        : `${left ? "-translate-x-full" : "translate-x-full"} lg:translate-x-0 lg:w-0 lg:border-transparent`;
  const Icon = left ? PanelLeft : PanelRight;
  return (
    <aside
      aria-label={title}
      className={`fixed top-0 bottom-0 z-30 w-72 overflow-hidden border-line bg-bg transition-all duration-300 ease-out lg:relative lg:z-auto lg:shrink-0 ${
        left ? "left-0 border-r" : "right-0 border-l"
      } ${state}`}
    >
      <div className="flex h-full w-72 flex-col">
        <div className="flex h-14 shrink-0 items-center justify-between px-4">
          <h2 className="text-sm font-semibold">{title}</h2>
          <button
            type="button"
            onClick={onToggle}
            className="rounded-lg p-1.5 text-muted hover:bg-surface hover:text-ink"
            aria-label={`Collapse ${title}`}
            title={`Collapse ${title}`}
          >
            <Icon size={17} />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-6">{children}</div>
      </div>
    </aside>
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
  const [sourcesOpen, setSourcesOpen] = useState<boolean | null>(null);
  const [guideOpen, setGuideOpen] = useState<boolean | null>(null);
  const [dragging, setDragging] = useState(false);
  const nextId = useRef(1);
  const scroller = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

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

  // Follow the conversation while it grows, unless the reader has scrolled up.
  const followBottom = useCallback(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, []);

  const busy = turns.some((t) => !t.result && !t.error);
  const selectedSources = sources.filter((s) => selected.has(s.filename));
  // Case studies come from the most recently added selected source.
  const focus = [...selectedSources].reverse().find((s) => s.guide_status !== "failed") ?? selectedSources.at(-1);
  const canAsk = selectedSources.length > 0 && !busy;

  // From the "not toggled yet" state, flip whatever the screen size currently shows.
  const toggle = (open: boolean | null, set: (v: boolean) => void) => () =>
    set(open === null ? !window.matchMedia(DESKTOP).matches : !open);

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

  function toggleSource(filename: string) {
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
    if (textarea.current) textarea.current.style.height = "";
    setTurns((prev) => [...prev, { id, question: q, strategy, generate, sourceCount: names.length, caseStudy }]);
    requestAnimationFrame(() => {
      if (scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight;
    });
    const update = (patch: Partial<Turn>) =>
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)));
    try {
      update({ result: await ask(q, strategy, generate, names) });
    } catch (e) {
      update({ error: (e as Error).message });
    }
  }

  const lastId = turns.at(-1)?.id;

  return (
    <div
      className="flex h-full"
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

      <SidePanel side="left" title="Sources" open={sourcesOpen} onToggle={toggle(sourcesOpen, setSourcesOpen)}>
        <SourcesPanel
          sources={sources}
          selected={selected}
          uploading={uploading}
          uploadError={uploadError}
          onToggle={toggleSource}
          onToggleAll={() =>
            setSelected(
              sources.every((s) => selected.has(s.filename)) ? new Set() : new Set(sources.map((s) => s.filename)),
            )
          }
          onPick={() => fileInput.current?.click()}
          onRemove={onRemove}
        />
      </SidePanel>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between px-3">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={toggle(sourcesOpen, setSourcesOpen)}
              className={`rounded-lg p-2 text-muted hover:bg-surface hover:text-ink ${sourcesOpen === null ? "lg:hidden" : sourcesOpen ? "hidden" : ""}`}
              aria-label="Show sources"
              title="Show sources"
            >
              <PanelLeft size={17} />
            </button>
            <span className="px-1 text-[15px] font-semibold tracking-tight">Grounded Notebook</span>
          </div>
          <div className="flex items-center gap-1">
            {turns.length > 0 && (
              <button
                type="button"
                onClick={() => setTurns([])}
                className="rounded-lg p-2 text-muted hover:bg-surface hover:text-ink"
                aria-label="New chat"
                title="New chat"
              >
                <SquarePen size={17} />
              </button>
            )}
            <button
              type="button"
              onClick={toggle(guideOpen, setGuideOpen)}
              className={`rounded-lg p-2 text-muted hover:bg-surface hover:text-ink ${guideOpen === null ? "lg:hidden" : guideOpen ? "hidden" : ""}`}
              aria-label="Show guide"
              title="Show guide"
            >
              <PanelRight size={17} />
            </button>
          </div>
        </header>

        <main ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-5 pt-6 pb-10">
            {loadError && <p className="mb-8 text-sm">{loadError}</p>}

            {sources.length === 0 ? (
              <div className="fade-up pt-10 text-center sm:pt-20">
                <h1 className="text-3xl font-semibold tracking-tight">Ask your documents</h1>
                <p className="mx-auto mt-3 max-w-md text-muted">
                  Answers come only from the files you add, every claim is cited, and it tells you when the answer
                  isn&apos;t there.
                </p>
                <button
                  type="button"
                  onClick={() => fileInput.current?.click()}
                  disabled={uploading !== null}
                  className={`mx-auto mt-10 flex w-full max-w-md flex-col items-center gap-2 rounded-3xl border border-dashed px-6 py-12 transition-colors disabled:opacity-60 ${
                    dragging ? "border-ink bg-surface" : "border-faint/60 hover:bg-surface"
                  }`}
                >
                  <Upload size={20} className={uploading ? "animate-pulse" : "text-muted"} />
                  <span className="font-medium">{uploading ? `Adding ${uploading}…` : "Add a source"}</span>
                  <span className="text-xs text-faint">PDF, DOCX, TXT or MD · or drop it anywhere</span>
                </button>
                {uploadError && <p className="mt-3 text-sm">{uploadError}</p>}
              </div>
            ) : turns.length === 0 ? (
              <div className="fade-up pt-10 sm:pt-16">
                <h1 className="text-center text-3xl font-semibold tracking-tight">
                  {selectedSources.length === 0 ? "Select a source to begin" : "What would you like to know?"}
                </h1>
                <p className="mt-2 text-center text-sm text-faint">
                  {selectedSources.length === 0
                    ? "Tick a file in Sources"
                    : `Answering from ${selectedSources.map((s) => s.filename).join(", ")}`}
                </p>
                {focus?.guide_status === "pending" && (
                  <p className="shimmer mt-10 text-center text-sm">Reading the document to suggest questions…</p>
                )}
                {focus && focus.guide_status !== "pending" && (
                  <div className="mt-10 grid gap-2 sm:grid-cols-2">
                    {focus.case_studies.map((c) => (
                      <button
                        key={c.kind}
                        type="button"
                        disabled={!canAsk}
                        onClick={() => run(c.question, c)}
                        className="rounded-2xl border border-line px-4 py-3 text-left transition-colors hover:bg-surface disabled:opacity-40"
                      >
                        <span className="block text-[11px] font-medium tracking-wider text-faint uppercase">{c.label}</span>
                        <span className="mt-1 block text-sm leading-snug">{c.question}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-10">
                {turns.map((turn) => (
                  <ChatTurn key={turn.id} turn={turn} animate={turn.id === lastId} onGrow={followBottom} />
                ))}
              </div>
            )}
          </div>
        </main>

        <footer className="relative shrink-0 px-4 pt-1 pb-3">
          {/* Soft fade so the conversation scrolls away under the composer instead of being cut. */}
          <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-t from-bg to-transparent" />
          <div className="mx-auto max-w-3xl">
            {turns.length > 0 && focus && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {focus.case_studies.map((c) => (
                  <button
                    key={c.kind}
                    type="button"
                    disabled={!canAsk}
                    title={`${c.question}\nExpected: ${c.expect}`}
                    onClick={() => run(c.question, c)}
                    className="rounded-full border border-line px-3 py-1 text-xs text-muted transition-colors hover:border-faint hover:text-ink disabled:opacity-40"
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                run(question);
              }}
              className="flex items-end gap-2 rounded-3xl border border-line bg-surface py-2 pr-2 pl-5 transition-colors focus-within:border-faint"
            >
              <textarea
                ref={textarea}
                rows={1}
                value={question}
                onChange={(e) => {
                  setQuestion(e.target.value);
                  e.target.style.height = "";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    run(question);
                  }
                }}
                disabled={selectedSources.length === 0}
                placeholder={
                  sources.length === 0
                    ? "Add a source to start"
                    : selectedSources.length === 0
                      ? "Select a source first"
                      : "Ask anything about your sources"
                }
                className="max-h-[200px] min-w-0 flex-1 resize-none bg-transparent py-1.5 leading-6 outline-none placeholder:text-faint"
                aria-label="Your question"
              />
              <button
                type="submit"
                disabled={!canAsk || !question.trim()}
                className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-ink text-bg transition-opacity disabled:opacity-20"
                aria-label="Send"
              >
                <ArrowUp size={18} />
              </button>
            </form>
            <p className="mt-2 text-center text-[11px] text-faint">
              Answers come only from your selected sources and can still be wrong. Check the citations.
            </p>
          </div>
        </footer>
      </div>

      <SidePanel side="right" title="Guide" open={guideOpen} onToggle={toggle(guideOpen, setGuideOpen)}>
        <GuidePanel
          meta={meta}
          guides={selectedSources}
          strategy={strategy}
          generate={generate}
          onStrategy={setStrategy}
          onGenerate={setGenerate}
        />
      </SidePanel>
    </div>
  );
}
