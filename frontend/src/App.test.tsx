import { StrictMode } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { RunRecord, StoredRun } from './api';

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
const completedRun: RunRecord = {
  trace_id: evidence.trace_id, state: 'completed', reason: 'published', terminal_stage: 'publish', latency_ms: 1000,
  stages: ['retrieve', 'context', 'generate', 'citation_check', 'verify', 'publication_check', 'publish'].map(name => ({ name, status: 'completed', latency_ms: 10 })),
};
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
  await act(async()=>done(json({...answered, run: completedRun})));
  expect(screen.queryByText('已核验的结论')).toBeNull();
  expect(screen.queryByText('运行记录')).toBeNull();
});

describe('completed request records', () => {
  it('shows a collapsible final record with stage timings and reuses the response trace', async () => {
    setupAnswer({ ...answered, run: completedRun });
    render(<App />); await submitAnswer();
    const summary = await screen.findByText('运行记录');
    const details = summary.closest('details')!;
    expect(details.open).toBe(false);
    fireEvent.click(summary);
    expect(details.open).toBe(true);
    const record = within(details);
    expect(record.getByText(/这是请求结束后返回的记录，不是实时进度/)).toBeTruthy();
    expect(record.getByText('published')).toBeTruthy();
    expect(record.getByText(`Trace ${evidence.trace_id}`)).toBeTruthy();
    expect(record.getByText('1,000 ms')).toBeTruthy();
    const stages = within(record.getByRole('table', { name: '运行阶段记录' }));
    expect(stages.getAllByText('已完成')).toHaveLength(7);
    expect(stages.getAllByText('10 ms')).toHaveLength(7);
  });

  it('explains a verification block and leaves unexecuted stages visible without exposing a draft', async () => {
    const run: RunRecord = { ...completedRun, state: 'blocked', reason: 'semantic_verification_failed', terminal_stage: 'verify', stages: completedRun.stages.map((stage, index) => ({ ...stage, status: index < 4 ? 'completed' : index === 4 ? 'stopped' : 'not_run', latency_ms: index > 4 ? null : 10 })) };
    setupAnswer({ ...answered, status: 'verification_failed', run });
    render(<App />); await submitAnswer();
    const details = (await screen.findByText('运行记录')).closest('details')!;
    fireEvent.click(within(details).getByText('运行记录'));
    const record = within(details);
    expect(record.getByText('语义支持核验未通过')).toBeTruthy();
    expect(record.getByText('semantic_verification_failed')).toBeTruthy();
    expect(record.getByText('在此停止')).toBeTruthy();
    expect(record.getAllByText('未执行')).toHaveLength(2);
    expect(record.getAllByText('—')).toHaveLength(2);
    expect(screen.queryByText('已核验的结论')).toBeNull();
  });

  it('displays an HTTP 409 record, clears it on retry, and accepts a legacy response', async () => {
    let attempts = 0;
    let finishRetry: (response: Response) => void = () => {};
    const run: RunRecord = { ...completedRun, state: 'blocked', reason: 'corpus_changed', terminal_stage: 'publication_check' };
    setupAnswer(() => {
      attempts++;
      return attempts === 1
        ? Promise.resolve(new Response(JSON.stringify({ detail: '语料已变化，请重试', run }), { status: 409 }))
        : new Promise<Response>(resolve => { finishRetry = resolve; });
    });
    render(<App />); await submitAnswer();
    await screen.findByText('语料已变化，请重试');
    expect(screen.getByText('corpus_changed')).toBeTruthy();
    expect(screen.queryByText('已核验的结论')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: '重试' }));
    expect(screen.queryByText('运行记录')).toBeNull();
    await act(async () => finishRetry(json(answered)));
    await screen.findByText('已核验的结论');
    expect(screen.queryByText('运行记录')).toBeNull();
  });

  it.each(['paper', 'task', 'question'] as const)('clears a completed record when changing the %s', async change => {
    setupAnswer({ ...answered, run: completedRun });
    render(<App />); await submitAnswer();
    await screen.findByText('运行记录');
    if (change === 'paper') fireEvent.click(screen.getByRole('button', { name: /Second paper title/ }));
    if (change === 'task') fireEvent.change(screen.getByLabelText('任务'), { target: { value: 'evidence' } });
    if (change === 'question') fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'new question' } });
    expect(screen.queryByText('运行记录')).toBeNull();
  });
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


const storedRun: StoredRun = {
  run_id: '0123456789abcdef0123456789abcdef', paper_id: 'p1', state: 'completed', reason: 'published',
  revision: 8, created_at: '2026-09-23T12:00:00Z', updated_at: '2026-09-23T12:01:00Z', cancel_requested: false,
  snapshot: { run: completedRun }, metadata: {},
};

describe('controlled answer execution', () => {
  it('sends a client run id and leaves repair disabled unless explicitly selected', async () => {
    const fetch = setupAnswer(answered); render(<App />); await submitAnswer();
    await screen.findByText('已核验的结论');
    let body = JSON.parse(String(fetch.mock.calls.find(call => call[0] === '/api/v1/answer')?.[1].body));
    expect(body.run_id).toMatch(/^[a-f0-9]{32}$/);
    expect(body.allow_repair).toBe(false);
    fireEvent.click(screen.getByText('实验设置'));
    fireEvent.click(screen.getByRole('checkbox', { name: '允许一次修复并重新核验' }));
    fireEvent.click(screen.getByRole('button', { name: '生成并核验回答' }));
    await screen.findByText('已核验的结论');
    const answers = fetch.mock.calls.filter(call => call[0] === '/api/v1/answer');
    body = JSON.parse(String(answers[1][1].body));
    expect(body.allow_repair).toBe(true);
    expect(body.run_id).not.toBe(JSON.parse(String(answers[0][1].body)).run_id);
  });

  it('requests server cancellation for the same run before stopping the wait', async () => {
    let signal: AbortSignal | null = null;
    let finishCancel: (response: Response) => void = () => {};
    const fetch = setupAnswer((options: RequestInit) => { signal = options.signal || null; return new Promise(() => {}); });
    const base = fetch.getMockImplementation()!;
    fetch.mockImplementation((url, options) => url.endsWith('/cancel') ? new Promise(resolve => { finishCancel = resolve; }) : base(url, options));
    render(<App />); await submitAnswer();
    const body = JSON.parse(String(fetch.mock.calls.find(call => call[0] === '/api/v1/answer')?.[1].body));
    fireEvent.click(screen.getByRole('button', { name: '取消生成' }));
    expect(fetch.mock.calls.some(call => call[0] === `/api/v1/runs/${body.run_id}/cancel` && call[1].method === 'POST')).toBe(true);
    expect(signal!.aborted).toBe(false);
    await act(async () => finishCancel(json({ ...storedRun, run_id: body.run_id, state: 'running', reason: '', cancel_requested: true, snapshot: {} })));
    expect(signal!.aborted).toBe(true);
    expect(await screen.findByText(/服务端已接受取消请求，已停止等待/)).toBeTruthy();
    expect(screen.queryByText('服务端已确认运行取消。')).toBeNull();
  });

  it.each([404, 503])('does not claim cancellation or stop waiting on a %s response', async status => {
    let signal: AbortSignal | null = null;
    const fetch = setupAnswer((options: RequestInit) => { signal = options.signal || null; return new Promise(() => {}); });
    const base = fetch.getMockImplementation()!;
    fetch.mockImplementation((url, options) => url.endsWith('/cancel') ? Promise.resolve(new Response(JSON.stringify({ detail: '取消请求失败' }), { status })) : base(url, options));
    render(<App />); await submitAnswer();
    fireEvent.click(screen.getByRole('button', { name: '取消生成' }));
    await screen.findByText(status === 404 ? /尚未找到运行记录，取消未获确认/ : /取消未获确认；仍在等待回答/);
    expect(signal!.aborted).toBe(false);
    expect(screen.getByRole('button', { name: '取消生成' })).toBeTruthy();
    expect(screen.queryByText('服务端已确认运行取消。')).toBeNull();
  });

  it('ignores a late cancel result after an answer completed and a new run started', async () => {
    const resolveAnswers: ((response: Response) => void)[] = [];
    const signals: AbortSignal[] = [];
    let finishCancel: (response: Response) => void = () => {};
    const fetch = setupAnswer((options: RequestInit) => { signals.push(options.signal!); return new Promise(resolve => resolveAnswers.push(resolve)); });
    const base = fetch.getMockImplementation()!;
    fetch.mockImplementation((url, options) => url.endsWith('/cancel') ? new Promise(resolve => { finishCancel = resolve; }) : base(url, options));
    render(<App />); await submitAnswer();
    fireEvent.click(screen.getByRole('button', { name: '取消生成' }));
    await act(async () => resolveAnswers[0](json(answered)));
    await screen.findByText('已核验的结论');
    fireEvent.click(screen.getByRole('button', { name: '生成并核验回答' }));
    await act(async () => finishCancel(json({ ...storedRun, state: 'running', cancel_requested: true, snapshot: {} })));
    expect(signals[1].aborted).toBe(false);
    expect(screen.getByRole('button', { name: '取消生成' })).toBeTruthy();
    expect(screen.queryByText(/服务端已接受取消请求/)).toBeNull();
  });

  it('renders budget limits and repeated attempts separately from reported usage', async () => {
    const run: RunRecord = { ...completedRun, state: 'failed', reason: 'call_budget_exceeded',
      limits: { deadline_seconds: 90, max_model_calls: 3, max_prompt_chars: 30000, max_completion_tokens: 2400, max_repairs: 1 },
      budget: { model_calls: 3, completion_tokens_reserved: 2400, reported_prompt_tokens: 2000, reported_completion_tokens: 200, usage_unknown_calls: 1 },
      attempts: [{ name: 'generate', status: 'completed', latency_ms: 30, attempt: 0 }, { name: 'generate', status: 'completed', latency_ms: 20, attempt: 1 }],
    };
    setupAnswer({ ...answered, status: 'model_unavailable', claims: [], run }); render(<App />); await submitAnswer();
    fireEvent.click(await screen.findByText('运行记录'));
    expect(screen.getByText('模型调用次数预算已用尽')).toBeTruthy();
    expect(screen.getByText(/有 1 次调用未返回用量，以上用量不完整/)).toBeTruthy();
    expect(within(screen.getByRole('table', { name: '阶段尝试记录' })).getAllByText('生成回答')).toHaveLength(2);
    expect(screen.queryByText('已核验的结论')).toBeNull();
  });
});

describe('run history and identity boundaries', () => {
  it('only loads history on demand and clears it when changing paper', async () => {
    const fetch = setupFetch(); const base = fetch.getMockImplementation()!;
    fetch.mockImplementation((url, options) => url.startsWith('/api/v1/runs?') ? Promise.resolve(json({ items: [storedRun] })) : url === `/api/v1/runs/${storedRun.run_id}` ? Promise.resolve(json(storedRun)) : base(url, options));
    render(<App />); await screen.findByRole('heading', { level: 1, name: papers[0].title });
    expect(fetch.mock.calls.some(call => call[0].startsWith('/api/v1/runs'))).toBe(false);
    fireEvent.click(screen.getByText('当前论文的最近运行'));
    fireEvent.click(screen.getByRole('button', { name: '刷新运行记录' }));
    fireEvent.click(await screen.findByRole('button', { name: `查看运行 ${storedRun.run_id}` }));
    await screen.findByText(/这是历史运行快照，不包含答案全文/);
    expect(fetch.mock.calls.some(call => call[0] === '/api/v1/runs?limit=20&paper_id=p1')).toBe(true);
    expect(screen.queryByText('已核验的结论')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Second paper title/ }));
    expect(screen.queryByText(/这是历史运行快照/)).toBeNull();
    expect(screen.queryByRole('button', { name: `查看运行 ${storedRun.run_id}` })).toBeNull();
  });

  it('aborts old requests and clears prior identity data before reloading with an in-memory credential', async () => {
    let finishAnswer: (response: Response) => void = () => {};
    let signal: AbortSignal | null = null;
    const fetch = setupAnswer((options: RequestInit) => { signal = options.signal || null; return new Promise(resolve => { finishAnswer = resolve; }); });
    const base = fetch.getMockImplementation()!;
    fetch.mockImplementation((url, options) => new Headers(options.headers).has('Authorization') ? new Promise(() => {}) : base(url, options));
    render(<App />); await submitAnswer();
    fireEvent.click(screen.getByText('访问凭证'));
    fireEvent.change(screen.getByLabelText('Bearer 凭证'), { target: { value: 'second-identity-token' } });
    fireEvent.click(screen.getByRole('button', { name: '应用凭证并重载' }));
    expect(signal!.aborted).toBe(true);
    expect(screen.queryByRole('heading', { level: 1, name: papers[0].title })).toBeNull();
    expect(screen.queryByText(papers[1].title)).toBeNull();
    expect((screen.getByLabelText('Bearer 凭证') as HTMLInputElement).value).toBe('');
    expect(Object.values(window.localStorage)).not.toContain('second-identity-token');
    expect(window.location.href).not.toContain('second-identity-token');
    await waitFor(() => expect(fetch.mock.calls.some(call => call[0] === '/api/v1/system' && new Headers(call[1].headers).get('Authorization') === 'Bearer second-identity-token')).toBe(true));
    await act(async () => finishAnswer(json(answered)));
    expect(screen.queryByText('已核验的结论')).toBeNull();
  });
});


it('can open the minimal just-created run before any stage snapshot exists', async () => {
  const firstRun: StoredRun = { ...storedRun, state: 'running', reason: null,
    snapshot: { run: { trace_id: storedRun.run_id, state: 'running', reason: null, terminal_stage: null, stages: [], attempts: [] } },
  };
  const fetch = setupFetch(); const base = fetch.getMockImplementation()!;
  fetch.mockImplementation((url, options) => url.startsWith('/api/v1/runs?') ? Promise.resolve(json({ items: [firstRun] })) : url === `/api/v1/runs/${storedRun.run_id}` ? Promise.resolve(json(firstRun)) : base(url, options));
  render(<App />); await screen.findByRole('heading', { level: 1, name: papers[0].title });
  fireEvent.click(screen.getByText('当前论文的最近运行'));
  fireEvent.click(screen.getByRole('button', { name: '刷新运行记录' }));
  fireEvent.click(await screen.findByRole('button', { name: `查看运行 ${storedRun.run_id}` }));
  fireEvent.click(await screen.findByText('运行记录'));
  expect(screen.getByText('运行尚未结束')).toBeTruthy();
  expect(screen.getByText('尚未开始')).toBeTruthy();
});

it('observes the submitted run while preserving the original answer delivery', async () => {
  let finish: (value: Response) => void = () => {};
  let ident = '';
  let signal: AbortSignal;
  const fetch = setupAnswer((options: RequestInit) => {
    ident = JSON.parse(String(options.body)).run_id; signal = options.signal!;
    return new Promise(resolve => { finish = resolve; });
  });
  const base = fetch.getMockImplementation()!;
  fetch.mockImplementation((url, options) => url.startsWith('/api/v1/runs/')
    ? Promise.resolve(json({ ...storedRun, run_id: ident, snapshot: { run: { ...completedRun, trace_id: ident } } }))
    : base(url, options));
  render(<App />); await submitAnswer();
  await screen.findByText(/正在接收回答；状态记录不会代替答案正文/);
  expect(screen.queryByText('已核验的结论')).toBeNull(); expect(signal!.aborted).toBe(false);
  await act(async () => finish(json({ ...answered, run: { ...completedRun, trace_id: ident } })));
  await screen.findByText('已核验的结论');
  expect(screen.queryByLabelText('当前运行状态')).toBeNull();
  expect(fetch.mock.calls.filter(([url]) => url === '/api/v1/answer')).toHaveLength(1);
});

it('continues observing after acknowledged cancellation and automatically displays its final reason', async () => {
  let ident = '';
  const fetch = setupAnswer((options: RequestInit) => { ident = JSON.parse(String(options.body)).run_id; return new Promise(() => {}); });
  const base = fetch.getMockImplementation()!;
  fetch.mockImplementation((url, options) => url.endsWith('/cancel')
    ? Promise.resolve(json({ ...storedRun, run_id: ident, state: 'running', cancel_requested: true, snapshot: {} }))
    : url.startsWith('/api/v1/runs/') ? Promise.resolve(json({ ...storedRun, run_id: ident, state: 'failed', reason: 'cancelled',
      snapshot: { run: { ...completedRun, trace_id: ident, state: 'failed', reason: 'cancelled' } } }))
    : base(url, options));
  render(<App />); await submitAnswer();
  fireEvent.click(screen.getByRole('button', { name: '取消生成' }));
  await screen.findByText(/服务端已接受取消请求/);
  await waitFor(() => expect(screen.getByRole('status').textContent).toBe('运行已取消'));
  expect(screen.queryByText(/服务端已接受取消请求/)).toBeNull();
  expect(screen.queryByText('已核验的结论')).toBeNull();
});

it('aborts status observation on paper switch and rejects a late old snapshot', async () => {
  let finish: (value: Response) => void = () => {};
  let signal: AbortSignal;
  let ident = '';
  const fetch = setupAnswer((options: RequestInit) => { ident = JSON.parse(String(options.body)).run_id; return new Promise(() => {}); });
  const base = fetch.getMockImplementation()!;
  fetch.mockImplementation((url, options) => url.startsWith('/api/v1/runs/')
    ? new Promise(resolve => { signal = options.signal!; finish = resolve; }) : base(url, options));
  render(<App />); await submitAnswer();
  await waitFor(() => expect(signal!).toBeDefined());
  fireEvent.click(screen.getByRole('button', { name: /Second paper title/ }));
  expect(signal!.aborted).toBe(true);
  await act(async () => finish(json({ ...storedRun, run_id: ident })));
  expect(screen.queryByLabelText('当前运行状态')).toBeNull();
  expect(screen.queryByText('运行记录')).toBeNull();
});

it('keeps observing after a gateway error that may hide an accepted run without resubmitting', async () => {
  let ident = '';
  const fetch = setupAnswer((options: RequestInit) => {
    ident = JSON.parse(String(options.body)).run_id;
    return Promise.resolve(new Response(JSON.stringify({ detail: '服务连接中断' }), { status: 502 }));
  });
  const base = fetch.getMockImplementation()!;
  fetch.mockImplementation((url, options) => url.startsWith('/api/v1/runs/')
    ? Promise.resolve(json({ ...storedRun, run_id: ident, state: 'interrupted', reason: 'server_restarted',
      snapshot: { run: { ...completedRun, trace_id: ident, state: 'interrupted', reason: 'server_restarted' } } }))
    : base(url, options));
  render(<App />); await submitAnswer();
  await screen.findByText('服务连接中断');
  await waitFor(() => expect(screen.getByRole('status').textContent).toContain('服务重启'));
  expect(fetch.mock.calls.filter(([url]) => url === '/api/v1/answer')).toHaveLength(1);
  expect(screen.queryByText('已核验的结论')).toBeNull();
});
