// Types mirror api/schemas.py. The API computes everything (titles, footnote
// numbers, gate marks); this app only draws it.

export type Status =
  | "answered"
  | "refused_by_gate"
  | "refused_by_model"
  | "retrieval_only"
  | "error";

export interface Passage {
  rank: number;
  title: string;
  source_filename: string;
  page_number: number;
  score: number;
  passed_gate: boolean | null;
  text: string;
}

export interface Source {
  number: number;
  title: string;
  source_filename: string;
  page_number: number;
  quote: string;
}

export interface Gate {
  passed: boolean;
  threshold: number;
  top_score: number | null;
  reason: string;
}

export interface QueryResponse {
  question: string;
  strategy: string;
  strategy_label: string;
  status: Status;
  claims: { text: string; sources: number[] }[];
  sources: Source[];
  abstain_reason: string | null;
  error: string | null;
  gate: Gate | null;
  passages: Passage[];
  seconds: number;
}

export interface Meta {
  papers: { title: string; chunks: number; uploaded: boolean }[];
  chunk_count: number;
  strategies: { key: string; label: string }[];
  examples: { label: string; question: string }[];
}

export interface UploadResponse {
  filename: string;
  chunks: number;
  chunk_count: number;
}

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error(`Cannot reach the API at ${API_URL}. Is \`make api\` running?`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(
      typeof detail === "string" ? detail : `Request failed (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}

export const getMeta = () => call<Meta>("/api/meta");

export const ask = (question: string, strategy: string, generate: boolean) =>
  call<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, strategy, generate }),
  });

export function upload(file: File) {
  const form = new FormData();
  form.append("file", file);
  return call<UploadResponse>("/api/upload", { method: "POST", body: form });
}
