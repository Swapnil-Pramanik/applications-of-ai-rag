# web/ — Next.js notebook UI

The frontend for *Grounded Notebook*: a NotebookLM-style page with a **Sources** panel (add, tick,
remove), a **chat** that answers only from the ticked sources, and a right-hand panel with the
notebook guide, settings and how it works. It only draws: every decision (retrieval, gate, answer,
footnote numbers, gate marks, case studies) comes from the FastAPI layer in `../api/`.

```bash
make api          # terminal 1, from the repo root: API on http://localhost:8000
make web          # terminal 2: this app on http://localhost:3000
```

The API URL defaults to `http://localhost:8000`; override it with `NEXT_PUBLIC_API_URL`. The
browser calls the API directly (the API allows CORS from `localhost:3000`), with no Next.js proxy, so
a slow Gemini call (up to ~45 s on the free tier) cannot hit a proxy timeout.

| File | Responsibility |
|---|---|
| `src/app/layout.tsx` | Inter font, metadata, `suppressHydrationWarning` on `<html>` (extension attributes) |
| `src/app/globals.css` | colour tokens (light + dark), Tailwind v4 theme, `fade-up` animation |
| `src/app/page.tsx` | renders `Notebook` |
| `src/lib/api.ts` | typed client: `getMeta`, `getSources`, `addSource`, `removeSource`, `ask`; types mirror `api/schemas.py` |
| `src/components/Notebook.tsx` | page state; top bar; Sources / Guide drawers; empty, welcome (summary + case-study list) and chat views; composer with a "Try" row; drop-anywhere upload; polls sources while a guide is pending |
| `src/components/SourcesPanel.tsx` | add, include / exclude, select all, remove, guide status |
| `src/components/ChatTurn.tsx` | one question: answer with superscript citations (click → quote inline), quiet refusal / error notes, pipeline line, expected behaviour, collapsible evidence |
| `src/components/Pipeline.tsx` | `stagesFor` (which of the five steps ran, where it stopped) and its one-line rendering |
| `src/components/Evidence.tsx` | top-5 passages with scores and a hairline bar against the gate threshold |
| `src/components/GuidePanel.tsx` | summaries of the selected sources, model / retriever / "write answers" settings, how it works |

## Design

Minimal: white (near-black in dark mode), neutral greys, one indigo accent (`--accent`), Inter
throughout, hairline borders instead of boxes. The chat is the page; Sources and Guide are drawers
from the top bar (overlay with a backdrop below `lg`). Refusals are a one-line amber note, not a
banner. Tokens live in `globals.css` (`--bg --surface --ink --muted --faint --line --accent
--accent-soft --good --warn --bad`); components use them via Tailwind (`bg-bg`, `text-muted`, …).
Icons: `lucide-react` (pinned).

Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4, `lucide-react` icons. No component library.
