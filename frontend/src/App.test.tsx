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

afterEach(() => { cleanup(); window.localStorage.clear(); vi.unstubAllGlobals(); });

describe('research session state', () => {
  it('sends the optional rerank flag and displays negative logits as rerank scores', async () => {
    let submitted: Record<string, unknown> = {};
    setupFetch(options => {
      submitted = JSON.parse(String(options.body));
      return Promise.resolve(json({ ...evidence, trace: { latency_ms: 100, retriever: 'hybrid', rerank_enabled: true },
        citations: [{ ...evidence.citations[0], score: -2.1234 }] }));
    });
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    expect((screen.getByRole('checkbox', { name: '模型重排' }) as HTMLInputElement).checked).toBe(false);
    fireEvent.click(screen.getByRole('checkbox', { name: '模型重排' }));
    fireEvent.change(screen.getByLabelText('检索方式'), { target: { value: 'hybrid' } });
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
    fireEvent.click(screen.getByRole('button', { name: '检索证据' }));
    await screen.findByText(evidence.citations[0].text);
    expect(submitted.rerank).toBe(true);
    expect(screen.getByText('-2.12')).toBeTruthy();
    expect(screen.getByText(/本次检索 · 混合 RRF \+ 模型重排/)).toBeTruthy();
    fireEvent.click(screen.getByRole('checkbox', { name: '模型重排' }));
    expect(screen.queryByText(evidence.citations[0].text)).toBeNull();
  });

  it('sends the selected retrieval mode and labels the returned RRF score', async () => {
    let submitted: Record<string, unknown> = {};
    setupFetch(options => {
      submitted = JSON.parse(String(options.body));
      return Promise.resolve(json({ ...evidence, trace: { latency_ms: 2, retriever: 'hybrid' },
        citations: [{ ...evidence.citations[0], score: 0.032258 }] }));
    });
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    fireEvent.change(screen.getByLabelText('检索方式'), { target: { value: 'hybrid' } });
    fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'method' } });
    fireEvent.click(screen.getByRole('button', { name: '检索证据' }));
    await screen.findByText(evidence.citations[0].text);
    expect(submitted.mode).toBe('hybrid');
    expect(screen.getByText('0.0323')).toBeTruthy();
    expect(screen.getByText(/本次检索 · 混合 RRF/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('检索方式'), { target: { value: 'dense' } });
    expect(screen.queryByText(evidence.citations[0].text)).toBeNull();
  });

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

  it('resizes the two panes with an accessible separator and remembers the width', async () => {
    setupFetch();
    render(<App />);
    await screen.findByRole('heading', { level: 1, name: papers[0].title });
    const separator = screen.getByRole('separator', { name: '调整论文库与正文区宽度' });
    const initial = Number(separator.getAttribute('aria-valuenow'));
    fireEvent.keyDown(separator, { key: 'ArrowRight' });
    await waitFor(() => expect(separator.getAttribute('aria-valuenow')).toBe(String(initial + 16)));
    expect(separator.parentElement?.style.getPropertyValue('--library-width')).toBe(`${initial + 16}px`);
    await waitFor(() => expect(window.localStorage.getItem('research-agent-library-width')).toBe(String(initial + 16)));
    fireEvent.doubleClick(separator);
    expect(separator.getAttribute('aria-valuenow')).toBe('285');
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

function setupAnswer(answer: unknown | ((options: RequestInit) => Promise<Response>)) {
  const fetch = vi.fn((url: string, options: RequestInit) => {
    if (url === '/api/v1/system') return Promise.resolve(json({ ...system, capabilities: { generation: true }, generation: {model:'deepseek-flash'} }));
    if (url === '/api/v1/ingestions') return Promise.resolve(json({items:[]}));
    if (url.startsWith('/api/v1/papers?')) return Promise.resolve(json({items:papers,total:2}));
    if (url === '/api/v1/answer') return typeof answer === 'function' ? answer(options) : Promise.resolve(json(answer));
    throw new Error(`Unexpected URL ${url}`);
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}
const answered = { ...evidence, mode:'grounded_answer', status:'answered', notice:'核验后发布',
  claims:[{text:'已核验的结论', evidence:[{chunk_id:'chunk1',quote:'This result'}]}],
  generation:{model_calls:2,latency_ms:1000,checks:{citation_integrity:'passed',semantic_support:'passed'}} };
async function submitAnswer() {
  await screen.findByRole('heading',{level:1,name:papers[0].title});
  fireEvent.change(screen.getByLabelText('任务'),{target:{value:'answer'}});
  fireEvent.change(screen.getByLabelText('输入检索问题'),{target:{value:'method'}});
  fireEvent.click(screen.getByRole('button',{name:'生成并核验回答'}));
}
it('publishes verified claims with quotations, and clears answers on task change',async()=>{
  const fetch=setupAnswer(answered); render(<App/>); await submitAnswer();
  await screen.findByText('已核验的结论');
  expect(screen.getByText('This result')).toBeTruthy();
  expect(fetch.mock.calls.some(c=>c[0]==='/api/v1/answer')).toBe(true);
  fireEvent.change(screen.getByLabelText('任务'),{target:{value:'evidence'}});
  expect(screen.queryByText('已核验的结论')).toBeNull();
});
it('never renders draft claims from a failed verification response',async()=>{
  setupAnswer({...answered,status:'verification_failed',notice:'草稿已拦截'}); render(<App/>); await submitAnswer();
  await screen.findByText('回答未通过核验');
  expect(screen.queryByText('已核验的结论')).toBeNull();
  expect(screen.getByText(evidence.citations[0].text)).toBeTruthy();
});
it('ignores late generated answers after switching the paper',async()=>{
  let done:(response:Response)=>void=()=>{};
  let signal:AbortSignal|null=null;
  setupAnswer((options:RequestInit)=>{signal=options.signal || null; return new Promise<Response>(resolve=>{done=resolve;});});
  render(<App/>); await submitAnswer();
  fireEvent.click(screen.getByRole('button',{name:/Second paper title/}));
  expect(signal!.aborted).toBe(true);
  await act(async()=>done(json(answered)));
  expect(screen.queryByText('已核验的结论')).toBeNull();
});

it('keeps fusion constant separate from output count and sends selected weights',async()=>{
  let submitted:Record<string,unknown>={};
  setupFetch(options=>{ submitted=JSON.parse(String(options.body)); return Promise.resolve(json({...evidence,trace:{...evidence.trace,retriever:'hybrid',rrf_constant:10,dense_weight:.75}}));});
  render(<App/>); await screen.findByRole('heading',{level:1,name:papers[0].title});
  expect(screen.queryByLabelText('RRF 平滑常数 K')).toBeNull();
  fireEvent.change(screen.getByLabelText('检索方式'),{target:{value:'hybrid'}});
  fireEvent.change(screen.getByLabelText('RRF 平滑常数 K'),{target:{value:'10'}});
  fireEvent.change(screen.getByLabelText('向量融合权重'),{target:{value:'0.75'}});
  fireEvent.change(screen.getByLabelText('输入检索问题'),{target:{value:'method'}});
  fireEvent.click(screen.getByRole('button',{name:'检索证据'}));
  await screen.findByText(evidence.citations[0].text);
  expect(submitted).toMatchObject({rrf_constant:10,dense_weight:.75,top_k:5});
  expect(screen.getByText(/K=10 · 向量权重 75%/)).toBeTruthy();
  fireEvent.change(screen.getByLabelText('RRF 平滑常数 K'),{target:{value:'100'}});
  expect(screen.queryByText(evidence.citations[0].text)).toBeNull();
});

it('sends the explicitly selected v2 strategy and English language',async()=>{
  let submitted:Record<string,unknown>={};
  setupAnswer((options:RequestInit)=>{submitted=JSON.parse(String(options.body));return Promise.resolve(json(answered));});
  render(<App/>);await screen.findByRole('heading',{level:1,name:papers[0].title});
  fireEvent.change(screen.getByLabelText('任务'),{target:{value:'answer'}});
  expect((screen.getByLabelText('回答策略') as HTMLSelectElement).value).toBe('v1');
  fireEvent.change(screen.getByLabelText('回答策略'),{target:{value:'v2'}});
  fireEvent.change(screen.getByLabelText('回答语言'),{target:{value:'en'}});
  fireEvent.change(screen.getByLabelText('输入检索问题'),{target:{value:'method'}});
  fireEvent.click(screen.getByRole('button',{name:'生成并核验回答'}));
  await screen.findByText('已核验的结论');
  expect(submitted).toMatchObject({profile:'v2',language:'en'});
});
