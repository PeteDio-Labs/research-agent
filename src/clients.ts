/**
 * HTTP clients for research-agent.
 * Routes upstream calls (web-search, blog RAG) through MC Backend so the
 * agent only needs LAN reachability to the cluster's NodePort.
 */

const MC_URL = process.env.MC_BACKEND_URL || 'http://localhost:3000';
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://192.168.50.59:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'petedio-writer';
const TIMEOUT_MS = 30_000;
const OLLAMA_TIMEOUT_MS = 120_000;

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  source?: string;
  score?: number;
}

export interface SearchResponse {
  query: string;
  provider: string;
  results: SearchResult[];
  metadata: {
    durationMs: number;
    fallbackUsed: boolean;
    providersAttempted: string[];
  };
}

export interface RagChunk {
  id: string;
  postId?: string;
  sourceType: string;
  sourceRef: string;
  chunkIndex: number;
  chunkText: string;
  similarity: number;
}

export interface RagResponse {
  results: RagChunk[];
  count: number;
}

export async function searchWeb(query: string, maxResults: number): Promise<SearchResponse> {
  const res = await fetch(`${MC_URL}/api/v1/web-search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, maxResults }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`MC web-search → ${res.status}${body ? ` :: ${body}` : ''}`);
  }
  return res.json() as Promise<SearchResponse>;
}

export async function queryRag(query: string, topK: number): Promise<RagResponse> {
  const res = await fetch(`${MC_URL}/api/v1/rag/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, topK }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`MC rag/query → ${res.status}${body ? ` :: ${body}` : ''}`);
  }
  return res.json() as Promise<RagResponse>;
}

export interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export async function ollamaChat(messages: OllamaChatMessage[], model = OLLAMA_MODEL): Promise<string> {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, stream: false }),
    signal: AbortSignal.timeout(OLLAMA_TIMEOUT_MS),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Ollama chat → ${res.status}${body ? ` :: ${body}` : ''}`);
  }
  const data = (await res.json()) as { message?: { content?: string } };
  return data.message?.content ?? '';
}
