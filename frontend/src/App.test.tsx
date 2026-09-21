import { StrictMode } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';

const papers = [
  { paper_id: 'p1', title: 'First paper title', abstract: 'First abstract', source: 'qasper', split: 'train', version: 'v1', arxiv_primary_category: 'cs.CL', research_direction: 'nlp', research_direction_label: '自然语言处理', arxiv_pdf_url: 'https://arxiv.org/pdf/2001.00001v1' },
  { paper_id: 'p2', title: 'Second paper title', abstract: 'Second abstract', source: 'qasper', split: 'train', version: 'v1', arxiv_primary_category: 'cs.IR', research_direction: 'information_retrieval', research_direction_label: '信息检索', arxiv_pdf_url: 'https://arxiv.org/pdf/2001.00002v1' },
];
const system = { stage: 'P2A', mode: 'postgres', database: { status: 'ready' }, object_store: { status: 'ready', provider: 'S3-compatible', bucket: 'papers' }, corpus: { papers: 2, paragraphs: 20, objects: 2 }, capabilities: { generation: false, pdf_upload: false } };
const evidence = { trace_id: 'trace1', query: 'method', paper_id: 'p1', top_k: 5, status: 'evidence_found', citations: [{ chunk_id: 'chunk1', paper_id: 'p1', title: papers[0].title, section_name: 'Methods', section_index: 0, paragraph_index: 0, text: 'This result belongs to the first paper.', source: 'qasper', version: 'v1', rank: 1, score: 2 }], notice: '仅返回原文。', trace: { latency_ms: 1 } };
const json = (body: unknown) => new Response(JSON.stringify(body));

function setupFetch(retrieve?: (options: RequestInit) => Promise<Response>) {
  const fetch = vi.fn((url: string, options: RequestInit) => {
    if (url === '/api/v1/system') return Promise.resolve(json(system));
    if (url === '/api/v1/ingestions') return Promise.resolve(json({ items: [] }));
    if (url.startsWith('/api/v1/papers?')) return Promise.resolve(json({ total: 2, items: papers }));
    if (url === '/api/v1/retrieve') return retrieve ? retrieve(options) : Promise.resolve(json(evidence));
    throw new Error(`Unexpected URL ${url}`);
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('research session state', () => {
  it('clears previous evidence and the question when selecting another paper', async () => {
    setupFetch(); render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
    fireEvent.click(screen.getByRole('button', { name: '检索证据' }));
    await screen.findByText(evidence.citations[0].text);
    fireEvent.click(screen.getByRole('button', { name: /Second paper title/ }));
    expect(screen.queryByText(evidence.citations[0].text)).toBeNull();
    expect((screen.getByLabelText('输入检索问题') as HTMLTextAreaElement).value).toBe('');
    expect(screen.getByRole('heading', { level: 1, name: papers[1].title })).toBeTruthy();
  });

  it('aborts an in-flight search and ignores its late response after switching papers', async () => {
    let resolveRequest: (response: Response) => void = () => {};
    let signal: AbortSignal | null = null;
    setupFetch(options => { signal = options.signal || null; return new Promise(resolve => { resolveRequest = resolve; }); });
    render(<App />); await screen.findByRole('heading', { level: 1, name: papers[0].title });
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
    fireEvent.click(screen.getByRole('button', { name: '检索证据' }));
    fireEvent.click(screen.getByRole('button', { name: /Second paper title/ }));
    expect(signal!.aborted).toBe(true);
    await act(async () => resolveRequest(json(evidence)));
    expect(screen.queryByText(evidence.citations[0].text)).toBeNull();
  });

  it('renders server failure and retries the same valid query', async () => {
    let attempts = 0;
    setupFetch(() => { attempts++; return Promise.resolve(attempts === 1 ? new Response(JSON.stringify({ detail: '存储服务暂不可用' }), { status: 503 }) : json(evidence)); });
    render(<App />); await screen.findByRole('heading', { level: 1, name: papers[0].title });
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
    fireEvent.click(screen.getByRole('button', { name: '检索证据' }));
    await screen.findByText('存储服务暂不可用');
    fireEvent.click(screen.getByRole('button', { name: '重试' }));
    await screen.findByText(evidence.citations[0].text);
    expect(attempts).toBe(2);
  });

  it('only enables submitting nonblank questions', async () => {
    setupFetch(); render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: '   ' } });
    expect((screen.getByRole('button', { name: '检索证据' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
    await waitFor(() => expect((screen.getByRole('button', { name: '检索证据' }) as HTMLButtonElement).disabled).toBe(false));
  });

  it('requests server-side sorting and resets to the first sorted paper', async () => {
    const fetch = setupFetch();
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    fireEvent.change(screen.getByLabelText('论文排序方式'), { target: { value: 'title_desc' } });
    await waitFor(() => expect(fetch.mock.calls.some(call =>
      String(call[0]).includes('sort=title_desc'))).toBe(true));
    expect(screen.getByLabelText('论文排序方式')).toHaveProperty('value', 'title_desc');
  });

  it('filters by direction before applying the selected sort and exposes the original PDF', async () => {
    const fetch = setupFetch();
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    expect(screen.getByRole('link', { name: '打开/下载原论文 PDF' })).toHaveProperty('href', papers[0].arxiv_pdf_url);
    fireEvent.change(screen.getByLabelText('论文研究方向'), { target: { value: 'nlp' } });
    fireEvent.change(screen.getByLabelText('论文排序方式'), { target: { value: 'submitted_newest' } });
    await waitFor(() => expect(fetch.mock.calls.some(call => {
      const url = String(call[0]);
      return url.includes('direction=nlp') && url.includes('sort=submitted_newest');
    })).toBe(true));
  });

  it('keeps the source context open under StrictMode effect cleanup', async () => {
    const originalShowModal = HTMLDialogElement.prototype.showModal;
    const originalClose = HTMLDialogElement.prototype.close;
    HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
    HTMLDialogElement.prototype.close = function () {
      this.removeAttribute('open');
      this.dispatchEvent(new Event('close'));
    };
    try {
      const fetch = setupFetch();
      fetch.mockImplementation((url: string) => {
        if (url.startsWith('/api/v1/papers/p1/paragraphs?')) {
          return Promise.resolve(json({ total: 1, items: [evidence.citations[0]] }));
        }
        if (url === '/api/v1/system') return Promise.resolve(json(system));
        if (url === '/api/v1/ingestions') return Promise.resolve(json({ items: [] }));
        if (url.startsWith('/api/v1/papers?')) return Promise.resolve(json({ total: 2, items: papers }));
        if (url === '/api/v1/retrieve') return Promise.resolve(json(evidence));
        throw new Error(`Unexpected URL ${url}`);
      });
      render(<StrictMode><App /></StrictMode>);
      await screen.findByRole('heading', { level: 1, name: papers[0].title });
      fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
      fireEvent.click(screen.getByRole('button', { name: '检索证据' }));
      await screen.findByText(evidence.citations[0].text);
      fireEvent.click(screen.getByRole('button', { name: '查看原文上下文' }));
      await screen.findByRole('heading', { name: '在原文中查看' });
      expect((screen.getByRole('dialog') as HTMLDialogElement).open).toBe(true);
      fireEvent.click(screen.getByRole('button', { name: '关闭原文' }));
      expect(screen.queryByRole('dialog')).toBeNull();
    } finally {
      HTMLDialogElement.prototype.showModal = originalShowModal;
      HTMLDialogElement.prototype.close = originalClose;
    }
  });

  it('keeps database failure visible when ingestion history also fails', async () => {
    const fetch = setupFetch();
    fetch.mockImplementation((url: string) => {
      if (url === '/api/v1/system') return Promise.resolve(json({ ...system, database: { status: 'unavailable' } }));
      if (url === '/api/v1/ingestions') return Promise.resolve(new Response(JSON.stringify({ detail: '任务记录不可用' }), { status: 503 }));
      return Promise.resolve(json({ total: 0, items: [] }));
    });
    render(<App />);
    await screen.findByRole('button', { name: /存储待检查/ });
    fireEvent.click(screen.getByRole('button', { name: '数据与存储' }));
    expect(screen.getByText('暂不可用')).toBeTruthy();
    expect(screen.getByText('任务记录不可用')).toBeTruthy();
    expect(screen.queryByText('存储已连接')).toBeNull();
  });
});
