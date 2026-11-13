"use client";

import { useEffect, useRef, useState } from "react";

import AnswerPanel from "@/components/AnswerPanel";
import EvidencePanel from "@/components/EvidencePanel";
import Pipeline, { stagesFor } from "@/components/Pipeline";
import { ask, getMeta, upload, type Meta, type QueryResponse } from "@/lib/api";

export default function AskApp() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState("hybrid+rerank");
  const [generate, setGenerate] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const [uploadMessage, setUploadMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [showCorpus, setShowCorpus] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getMeta().then(setMeta, (e: Error) => setMetaError(e.message));
  }, []);

  async function run(q: string) {
    const text = q.trim();
    if (!text || loading) return;
    setQuestion(text);
    setLoading(true);
    setRequestError(null);
    setActiveSource(null);
    try {
      setResult(await ask(text, strategy, generate));
    } catch (e) {
      setResult(null);
      setRequestError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function onUpload(file: File) {
    setUploading(true);
    setUploadMessage(null);
    try {
      const done = await upload(file);
      setUploadMessage({ ok: true, text: `${done.filename} added: ${done.chunks} ${done.chunks === 1 ? "passage is" : "passages are"} now searchable.` });
      setMeta(await getMeta());
    } catch (e) {
      setUploadMessage({ ok: false, text: (e as Error).message });
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const stages = stagesFor(result && !loading ? result.strategy : strategy, generate, result, loading);
  const paperCount = meta ? meta.papers.filter((p) => !p.uploaded).length : 18;
  const uploadCount = meta ? meta.papers.length - paperCount : 0;

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 sm:px-6">
      <header className="flex items-center justify-between py-5">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent font-serif text-lg font-bold text-white">
            A
          </div>
          <span className="font-semibold">Ask the Papers</span>
        </div>
        <button
          type="button"
          onClick={() => setShowCorpus(!showCorpus)}
          className="rounded-full border border-line bg-card px-3 py-1.5 text-sm text-muted hover:text-ink"
          aria-expanded={showCorpus}
        >
          {paperCount} papers{uploadCount > 0 && ` + ${uploadCount} uploaded`}
          <span className="hidden sm:inline"> · {(meta?.chunk_count ?? 0).toLocaleString()} passages</span>
        </button>
      </header>

      {showCorpus && meta && (
        <div className="rise mb-6 rounded-2xl border border-line bg-card p-5">
          <h2 className="font-semibold">Corpus</h2>
          <ul className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2 lg:grid-cols-3">
            {meta.papers.map((p) => (
              <li key={p.title} className="flex min-w-0 justify-between gap-2">
                <span className="min-w-0 truncate">
                  {p.title}
                  {p.uploaded && <span className="ml-1 rounded bg-accent-soft px-1 text-xs text-accent">uploaded</span>}
                </span>
                <span className="shrink-0 text-muted">{p.chunks}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <section className="pt-6 pb-8 text-center">
        <h1 className="font-serif text-4xl font-semibold tracking-tight sm:text-5xl">
          Ask questions of landmark ML papers.
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-muted">
          Every claim cites a paper, page and passage, and the system says so when the papers
          don&apos;t contain the answer.
        </p>
      </section>

      {metaError && (
        <div className="mb-4 rounded-xl border border-bad/40 bg-bad-soft p-4 text-sm text-bad">{metaError}</div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(question);
        }}
        className="rounded-2xl border border-line bg-card p-3 shadow-sm focus-within:border-accent"
      >
        <div className="flex gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. How does LoRA initialise the A and B matrices?"
            className="min-w-0 flex-1 bg-transparent px-3 py-3 text-lg outline-none placeholder:text-muted/70"
            aria-label="Your question"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="rounded-xl bg-accent px-6 font-semibold text-white transition-opacity disabled:opacity-40"
          >
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-3 border-t border-line px-2 pt-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-muted">Retriever</span>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="rounded-lg border border-line bg-card px-2 py-1"
            >
              {(meta?.strategies ?? [{ key: "hybrid+rerank", label: "Hybrid + rerank (default)" }]).map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex cursor-pointer items-center gap-2">
            <button
              type="button"
              role="switch"
              aria-checked={generate}
              onClick={() => setGenerate(!generate)}
              className={`relative h-6 w-11 rounded-full transition-colors ${generate ? "bg-accent" : "bg-line"}`}
            >
              <span
                className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${generate ? "translate-x-5" : ""}`}
              />
            </button>
            <span>
              Generate with Gemini{" "}
              <span className="text-muted">{generate ? "" : "(off: retrieval only, no quota)"}</span>
            </span>
          </label>

          <div className="ml-auto">
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.txt,.md"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
            />
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
              className="rounded-lg border border-line px-3 py-1 hover:border-accent hover:text-accent disabled:opacity-50"
            >
              {uploading ? "Indexing…" : "+ Upload a document"}
            </button>
          </div>
        </div>
      </form>

      {uploadMessage && (
        <p className={`mt-2 text-sm ${uploadMessage.ok ? "text-good" : "text-bad"}`}>{uploadMessage.text}</p>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-sm text-muted">Try:</span>
        {meta?.examples.map((ex) => (
          <button
            key={ex.label}
            type="button"
            title={ex.question}
            disabled={loading}
            onClick={() => run(ex.question)}
            className="rounded-full border border-line bg-card px-3 py-1 text-sm hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {ex.label}
          </button>
        ))}
      </div>

      <div className="mt-8">
        <Pipeline stages={stages} />
      </div>

      {requestError && (
        <div className="mt-6 rounded-xl border border-bad/40 bg-bad-soft p-4 text-sm text-bad">{requestError}</div>
      )}

      {result && !loading && (
        <>
          <p className="rise mt-8 font-serif text-xl">
            <span className="text-muted">Q.</span> {result.question}
          </p>
          <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-5">
            <div className="min-w-0 lg:col-span-3">
              <AnswerPanel result={result} activeSource={activeSource} onSource={setActiveSource} />
            </div>
            <div className="min-w-0 lg:col-span-2">
              <EvidencePanel result={result} />
            </div>
          </div>
          <p className="mt-4 text-right text-xs text-muted">
            {result.strategy_label} · {result.seconds.toFixed(2)}s
          </p>
        </>
      )}

      {!result && !loading && !requestError && (
        <p className="mt-10 text-center text-sm text-muted">
          Ask a question or pick an example. The five steps above light up as the pipeline runs.
        </p>
      )}
    </div>
  );
}
