export interface Paper {
  paper_id: string;
  title: string;
  abstract: string;
  split: string;
  source: string;
  version: string;
  arxiv_submitted_at?: string | null;
  arxiv_primary_category?: string | null;
  arxiv_categories?: string[];
  arxiv_pdf_url?: string | null;
  research_direction?: string | null;
  research_direction_label?: string | null;
  journal_ref?: string | null;
  ccf_venue?: string | null;
  ccf_level?: 'A' | 'B' | 'C' | null;
  ccf_catalog_url?: string | null;
}

export interface Paragraph {
  chunk_id: string;
  paper_id: string;
  title: string;
  section_name: string;
  section_index: number;
  paragraph_index: number;
  text: string;
  source: string;
  version: string;
  text_sha256?: string;
}

export interface Citation extends Paragraph { rank: number; score: number }
export interface Page<T> { total: number; items: T[] }
export interface Retrieval {
  trace_id: string;
  query: string;
  paper_id: string;
  top_k: number;
  status: 'evidence_found' | 'no_lexical_match';
  citations: Citation[];
  notice: string;
  trace: { retriever: string; k1: number; b: number; corpus_paragraphs: number; paper_paragraphs: number; returned: number; latency_ms: number; model_calls: number };
}

export interface System {
  stage: string;
  mode: string;
  database: { status: string };
  object_store: { status: string; provider: string; bucket: string };
  corpus: { papers: number; paragraphs: number; objects: number };
  capabilities: { generation: boolean; pdf_upload: boolean };
}

export interface Ingestion {
  id: string;
  status: string;
  created_at: string;
  finished_at?: string | null;
  papers: number;
  paragraphs: number;
  objects: number;
  error?: string | null;
  reused: boolean;
}

export function isAbort(error: unknown): boolean {
  // DOMException is not an Error in every browser realm (or in jsdom).
  return typeof error === 'object' && error !== null && 'name' in error && error.name === 'AbortError';
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '请求未能完成，请稍后重试。';
}

async function checkedFetch(path: string, options?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new Error('无法连接后端服务，请确认服务已启动后重试。');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: unknown } | null;
    const detail = typeof body?.detail === 'string' ? body.detail : null;
    throw new Error(detail || `请求失败（HTTP ${response.status}），请稍后重试。`);
  }
  return response;
}

export async function request<T>(path: string, signal?: AbortSignal, body?: unknown): Promise<T> {
  const response = await checkedFetch(path, {
    signal,
    ...(body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }),
  });
  return response.json() as Promise<T>;
}

export async function loadContext(paperId: string, chunkId: string, signal: AbortSignal): Promise<Paragraph[]> {
  const paragraphs: Paragraph[] = [];
  let offset = 0;
  let total = Infinity;
  while (offset < total) {
    const page = await request<Page<Paragraph>>(`/api/v1/papers/${encodeURIComponent(paperId)}/paragraphs?limit=100&offset=${offset}`, signal);
    paragraphs.push(...page.items);
    total = page.total;
    if (!page.items.length) break;
    offset += page.items.length;
    const found = paragraphs.findIndex(item => item.chunk_id === chunkId);
    if (found >= 0 && (found < paragraphs.length - 1 || offset >= total)) {
      return paragraphs.slice(Math.max(0, found - 1), found + 2);
    }
  }
  throw new Error('当前论文版本中未找到这条证据，请重新检索后再试。');
}

export async function downloadSource(paperId: string, signal: AbortSignal): Promise<void> {
  const response = await checkedFetch(`/api/v1/papers/${encodeURIComponent(paperId)}/source`, { signal });
  const file = await response.blob();
  if (signal.aborted) return;
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${paperId.replace(/[^a-zA-Z0-9_.-]/g, '_')}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
