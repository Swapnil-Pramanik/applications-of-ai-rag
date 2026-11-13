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
| `src/app/layout.tsx` | fonts (Inter, Newsreader), metadata |
| `src/app/globals.css` | colour tokens (light + dark), Tailwind v4 theme |
| `src/app/page.tsx` | renders `Notebook` |
| `src/lib/api.ts` | typed client: `getMeta`, `getSources`, `addSource`, `removeSource`, `ask`; types mirror `api/schemas.py` |
| `src/components/Notebook.tsx` | page state and three-column layout; polls sources while a guide is pending; runs questions and case studies against the ticked sources |
| `src/components/SourcesPanel.tsx` | add (click or drop), tick / select all, remove, guide status |
| `src/components/ChatTurn.tsx` | one question + answer: pipeline strip, expected behaviour for case studies, cited claims, refusal / error banners, collapsible evidence |
| `src/components/Pipeline.tsx` | `stagesFor` (which of the five steps ran, where it stopped) and the pill row |
| `src/components/Evidence.tsx` | top-5 passages, scores and a bar against the gate threshold |
| `src/components/StudioPanel.tsx` | clicked citation, notebook guide (summaries), retriever + Gemini switch, how it works |

Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4. No component library.
