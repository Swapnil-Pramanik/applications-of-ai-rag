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
| `src/app/globals.css` | black-and-white tokens (light + dark), Tailwind v4 theme, `fade-up`, `caret` (typing cursor) and `shimmer` (thinking) |
| `src/app/page.tsx` | renders `Notebook` |
| `src/lib/useTypewriter.ts` | reveals an answer character by character (~2.5 s max; instant with reduced motion). Presentational only: the API returns the full answer |
| `src/lib/api.ts` | typed client: `getMeta`, `getSources`, `addSource`, `removeSource`, `ask`; types mirror `api/schemas.py` |
| `src/components/Notebook.tsx` | page state; Sources and Guide side columns (open on desktop at first load, collapsible; slide-over on phones); empty, welcome (case-study cards) and chat views; ChatGPT-style composer (Enter sends, Shift+Enter new line, auto-grow) with case-study chips; drop-anywhere upload; follows the conversation while it types |
| `src/components/SourcesPanel.tsx` | add, include / exclude, select all, remove, guide status |
| `src/components/ChatTurn.tsx` | one exchange: user bubble (right), assistant reply (left) with a thinking shimmer, typed text, superscript citations (click → quote inline), refusal / error replies, then pipeline line, expected behaviour and collapsible evidence |
| `src/components/Pipeline.tsx` | `stagesFor` (which of the five steps ran, where it stopped) and its one-line rendering |
| `src/components/Evidence.tsx` | top-5 passages with scores and a hairline bar against the gate threshold |
| `src/components/GuidePanel.tsx` | summaries of the selected sources, model / retriever / "write answers" settings, how it works |

## Design

Chat-first and **black and white only**: white (near-black in dark mode), greys, no accent colour.
Status is carried by icons and weight, not colour (the `good/warn/bad/accent` tokens are greys).
Sources (left) and Guide (right) are side columns, open at first load on desktop and collapsible
from their headers or the top bar; on phones they start closed and slide over the chat. The chat
reads like ChatGPT/Claude: grey user bubbles on the right, assistant replies on the left, typed
out, with citations, the pipeline line and evidence appearing once the typing finishes. Tokens
live in `globals.css` (`--bg --surface --ink --muted --faint --line …`). Icons: `lucide-react`
(pinned).

Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4, `lucide-react` icons. No component library.
