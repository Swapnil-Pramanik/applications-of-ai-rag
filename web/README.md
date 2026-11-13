# web/ — Next.js presentation UI

The frontend for *Ask the Papers*. It only draws: every decision (retrieval, gate, answer,
footnote numbers, gate marks) comes from the FastAPI layer in `../api/`, which wraps the same
`DemoService` the Streamlit demo and the eval harness use.

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
| `src/app/page.tsx` | renders `AskApp` |
| `src/lib/api.ts` | typed client: `getMeta`, `ask`, `upload`; types mirror `api/schemas.py` |
| `src/components/AskApp.tsx` | page state: question box, retriever, Gemini toggle, examples, upload, corpus list |
| `src/components/Pipeline.tsx` | the five-step strip (Retrieve → Fuse → Rerank → Gate → Answer), lit per result |
| `src/components/AnswerPanel.tsx` | cited claims with hoverable footnotes, sources, refusal / error banners |
| `src/components/EvidencePanel.tsx` | top-5 passages, scores and a bar against the gate threshold |

Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4. No component library.
