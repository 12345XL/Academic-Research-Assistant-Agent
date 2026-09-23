import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, FormEvent, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import Icon from './Icon';
import { ApiError, downloadSource, errorMessage, isAbort, loadContext, request } from './api';
import type { Citation, Ingestion, Page, Paper, Paragraph, Retrieval, RunRecord, System } from './api';

const PAGE_SIZE = 12;
const LIBRARY_WIDTH_KEY = 'research-agent-library-width';
const MIN_LIBRARY_WIDTH = 230;
const MAX_LIBRARY_WIDTH = 560;
const MIN_RESEARCH_WIDTH = 440;
const RESIZE_HANDLE_WIDTH = 8;
const number = (value: number | undefined) => value === undefined ? '—' : value.toLocaleString('zh-CN');
const date = (value?: string | null) => value ? new Date(value.replace(' ', 'T')).toLocaleString('zh-CN', { hour12: false }) : '—';
const paperDate = (value?: string | null) => value ? new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', timeZone: 'UTC' }).format(new Date(value)) : '日期未知';
const SORT_LABELS = {
  id_asc: '论文编号 ↑', id_desc: '论文编号 ↓',
  submitted_newest: 'arXiv 提交时间：新到旧', submitted_oldest: 'arXiv 提交时间：旧到新',
  ccf_best: 'CCF 级别：A → 未分级', title_asc: '标题 A → Z', title_desc: '标题 Z → A',
} as const;
type PaperSort = keyof typeof SORT_LABELS;
const DIRECTION_LABELS = {
  all: '全部方向', nlp: '自然语言处理', machine_learning: '机器学习',
  information_retrieval: '信息检索', artificial_intelligence: '人工智能',
  speech_audio: '语音与音频', computer_vision_multimedia: '视觉与多媒体',
  social_computing: '社会计算', human_computer_interaction: '人机交互',
  robotics: '机器人', other: '其他方向',
} as const;
type ResearchDirection = keyof typeof DIRECTION_LABELS;

function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="error-notice" role="alert"><Icon name="info" /><span>{message}</span>{retry && <button onClick={retry}>重试</button>}</div>;
}

const RUN_STATES: Record<RunRecord['state'], string> = {
  completed: '已完成', abstained: '未作答', blocked: '已拦截', failed: '运行失败',
};
const RUN_STAGES: Record<string, string> = {
  retrieve: '检索证据', context: '组装上下文', generate: '生成回答',
  citation_check: '检查引用', verify: '核验语义支持', publication_check: '检查发布条件', publish: '发布回答',
};
const STAGE_STATES: Record<RunRecord['stages'][number]['status'], string> = {
  completed: '已完成', stopped: '在此停止', failed: '失败', not_run: '未执行',
};
const RUN_REASONS: Record<string, string> = {
  published: '回答已通过当前检查并发布', generation_not_configured: '生成模型尚未配置',
  no_retrieved_evidence: '检索未找到可用证据', context_budget_excluded_all: '上下文预算未容纳任何证据',
  generator_abstained: '生成模型判断证据不足，未作答', citation_invalid: '引用完整性检查未通过',
  semantic_verification_failed: '语义支持核验未通过', provider_timeout: '模型服务请求超时',
  provider_http_error: '模型服务返回错误', provider_connection_error: '无法连接模型服务',
  invalid_output: '模型输出格式不符合要求', incomplete_output: '模型输出不完整', model_refused: '模型拒绝回答',
  corpus_changed: '运行期间论文语料发生变化', evidence_integrity_failed: '原文证据完整性检查未通过',
  vector_unavailable: '向量检索暂不可用', reranker_unavailable: '重排模型暂不可用',
  paper_not_found: '未找到当前论文', invalid_request: '请求参数无效',
  dependency_unavailable: '依赖服务暂不可用', internal_error: '运行出现内部错误',
};
const runDuration = (value: number | null) => value === null ? '—' : `${value.toLocaleString('zh-CN', { maximumFractionDigits: 1 })} ms`;

function RunRecordView({ run }: { run: RunRecord }) {
  return <details className="run-record">
    <summary><span>运行记录</span><span className={`run-state ${run.state}`}>{RUN_STATES[run.state]}</span><span>{runDuration(run.latency_ms)}</span></summary>
    <div className="run-record-body">
      <p className="run-record-hint">这是请求结束后返回的记录，不是实时进度。记录反映执行与检查结果，回答质量仍需对照原文判断。</p>
      <p className="run-reason">{RUN_REASONS[run.reason] || '本次运行已结束，请查看原因码'}<code>{run.reason}</code></p>
      <dl className="run-meta"><div><dt>最终状态</dt><dd>{RUN_STATES[run.state]} <code>{run.state}</code></dd></div><div><dt>结束阶段</dt><dd>{RUN_STAGES[run.terminal_stage] || run.terminal_stage} <code>{run.terminal_stage}</code></dd></div><div><dt>调用记录</dt><dd className="mono">Trace {run.trace_id}</dd></div></dl>
      <div className="table-scroll"><table className="run-stages" aria-label="运行阶段记录"><thead><tr><th scope="col">阶段</th><th scope="col">结果</th><th scope="col">耗时</th></tr></thead><tbody>{run.stages.map(stage => <tr key={stage.name}><th scope="row">{RUN_STAGES[stage.name] || stage.name}<code>{stage.name}</code></th><td>{STAGE_STATES[stage.status]}</td><td>{runDuration(stage.latency_ms)}</td></tr>)}</tbody></table></div>
    </div>
  </details>;
}

function ContextDialog({ citation, paragraphs, loading, error, close, retry }: {
  citation: Citation; paragraphs: Paragraph[]; loading: boolean; error: string;
  close: () => void; retry: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current; dialog?.showModal(); return () => dialog?.close(); }, []);
  return <dialog ref={ref} className="context-dialog" onCancel={event => { event.preventDefault(); close(); }} onClick={event => { if (event.target === event.currentTarget) close(); }} aria-labelledby="context-heading">
    <div className="dialog-shell">
      <header className="dialog-header"><div><p className="eyebrow">EVIDENCE CONTEXT</p><h2 id="context-heading">在原文中查看</h2></div><button className="icon-button" aria-label="关闭原文" onClick={close}><Icon name="close" /></button></header>
      <p className="dialog-description">展示命中段落及相邻段落，保留论文中的顺序。</p>
      {loading && <p className="loading-line" role="status">正在读取原文…</p>}
      {error && <ErrorNotice message={error} retry={retry} />}
      <div className="context-paragraphs">{paragraphs.map(paragraph => <article key={paragraph.chunk_id} className={paragraph.chunk_id === citation.chunk_id ? 'context-paragraph matched' : 'context-paragraph'}>
        <div className="context-label"><span>{paragraph.section_name || '未命名章节'}</span>{paragraph.chunk_id === citation.chunk_id && <span className="tag green">命中证据</span>}</div>
        <p lang="en">{paragraph.text}</p><span className="mono subtle">{paragraph.chunk_id}</span>
      </article>)}</div>
    </div>
  </dialog>;
}

function StorageView({ system, jobs, loading, error, jobsError, refresh }: {
  system: System | null; jobs: Ingestion[]; loading: boolean; error: string; jobsError: string; refresh: () => void;
}) {
  const validCounts = system && (system.mode === 'files' || system.database.status === 'ready');
  const labels: Record<string, string> = { ready: '连接正常', unavailable: '暂不可用', not_configured: '未接入', completed: '已完成', succeeded: '已完成', running: '处理中', failed: '失败', pending: '等待中' };
  return <main className="storage-view">
    <div className="section-heading"><div><p className="eyebrow">DATA & STORAGE</p><h1>每一条证据，都有出处。</h1><p>查看论文语料、持久化存储和实际导入记录。</p></div><button className="secondary-button" onClick={refresh} disabled={loading}><Icon name="refresh" className={loading ? 'spin' : ''} />刷新状态</button></div>
    {error && <ErrorNotice message={error} retry={refresh} />}
    <div className="stat-grid">
      {[['论文', validCounts ? system.corpus.papers : undefined, '已导入的论文记录', 'book'], ['证据段落', validCounts ? system.corpus.paragraphs : undefined, '可检索的正文段落', 'layers'], ['存储对象', validCounts ? system.corpus.objects : undefined, '数据库登记的文件对象', 'file']].map(([label, count, note, icon]) => <article className="stat-card" key={String(label)}><div className="stat-label"><span>{label}</span><Icon name={icon as 'book' | 'layers' | 'file'} /></div><strong>{number(count as number | undefined)}</strong><p>{note}</p></article>)}
    </div>
    <div className="storage-grid">
      <article className="storage-card"><div className="storage-card-heading"><span className="feature-icon"><Icon name="database" /></span><div><h2>结构化数据</h2><p>{system?.mode === 'files' ? 'JSONL 文件模式' : 'PostgreSQL'}</p></div><span className={`status-tag ${system?.database.status === 'ready' ? 'ok' : ''}`}><i />{system ? labels[system.database.status] || system.database.status : '读取中'}</span></div><p className="storage-description">保存论文、版本、段落和文件关联。检索结果回到数据库读取原文，使证据可以定位与核对。</p><div className="storage-foot"><span>当前运行模式</span><code>{system?.mode || '—'}</code></div></article>
      <article className="storage-card"><div className="storage-card-heading"><span className="feature-icon"><Icon name="layers" /></span><div><h2>原文与文件</h2><p>{system?.object_store.provider || '对象存储'}</p></div><span className={`status-tag ${system?.object_store.status === 'ready' ? 'ok' : ''}`}><i />{system ? labels[system.object_store.status] || system.object_store.status : '读取中'}</span></div><p className="storage-description">保存结构化论文正文文件。下载时检查内容哈希；QASPER 标准答案和评测标签不进入在线原文。</p><div className="storage-foot"><span>Bucket</span><code>{system?.object_store.bucket || '—'}</code></div></article>
    </div>
    <section className="jobs-section"><div className="jobs-heading"><div><h2>导入记录</h2><p>最近 20 次任务 · 来自后端实际运行记录</p></div><span className="tag">{jobs.length} 条记录</span></div>
      {jobsError && <ErrorNotice message={jobsError} retry={refresh} />}
      <div className="table-scroll"><table><thead><tr><th>任务 / 开始时间</th><th>状态</th><th>论文</th><th>段落</th><th>对象</th><th>完成时间</th></tr></thead><tbody>{jobs.map(job => <tr key={job.id}><td><code className="job-id" title={job.id}>{job.id.slice(0, 12)}</code><span className="cell-subtitle">{date(job.created_at)}</span>{job.error && <span className="job-error">{job.error}</span>}</td><td><span className={`status-tag ${['completed', 'succeeded'].includes(job.status) ? 'ok' : ''}`}>{labels[job.status] || job.status}</span>{job.reused && <span className="cell-subtitle">复用已有导入</span>}</td><td>{number(job.papers)}</td><td>{number(job.paragraphs)}</td><td>{number(job.objects)}</td><td className="subtle">{date(job.finished_at)}</td></tr>)}</tbody></table></div>
      {!jobs.length && <div className="table-empty">{loading ? '正在读取导入记录…' : jobsError ? '导入记录暂不可用，请重试。' : '暂无导入记录。完成一次数据导入后，会在这里显示。'}</div>}
    </section>
    <p className="storage-note"><Icon name="info" />当前工作台使用公共 QASPER 语料；个人 PDF 上传与解析将在后续阶段接入。</p>
  </main>;
}

export default function App() {
  const workbenchRef = useRef<HTMLDivElement>(null);
  const resizingRef = useRef(false);
  const [resizingPane, setResizingPane] = useState(false);
  const [libraryWidth, setLibraryWidth] = useState(() => {
    const fallback = window.innerWidth <= 1100 ? 285 : 326;
    try {
      const saved = Number(window.localStorage.getItem(LIBRARY_WIDTH_KEY));
      return Number.isFinite(saved) && saved >= MIN_LIBRARY_WIDTH && saved <= MAX_LIBRARY_WIDTH ? saved : fallback;
    } catch { return fallback; }
  });
  const [tab, setTab] = useState<'research' | 'storage'>('research');
  const [system, setSystem] = useState<System | null>(null);
  const [jobs, setJobs] = useState<Ingestion[]>([]);
  const [systemLoading, setSystemLoading] = useState(true);
  const [systemError, setSystemError] = useState('');
  const [jobsError, setJobsError] = useState('');
  const [systemRevision, setSystemRevision] = useState(0);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('');
  const [direction, setDirection] = useState<ResearchDirection>('all');
  const [sort, setSort] = useState<PaperSort>('id_asc');
  const [offset, setOffset] = useState(0);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [total, setTotal] = useState(0);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState('');
  const [listRevision, setListRevision] = useState(0);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [retrievalMode, setRetrievalMode] = useState('bm25');
  const [rerank, setRerank] = useState(false);
  const [rrfConstant, setRrfConstant] = useState(60);
  const [denseWeight, setDenseWeight] = useState(0.5);
  const [answerProfile, setAnswerProfile] = useState('v1');
  const [answerLanguage, setAnswerLanguage] = useState('zh');
  const [answerMode, setAnswerMode] = useState(false);
  const [result, setResult] = useState<Retrieval | null>(null);
  const [run, setRun] = useState<RunRecord | null>(null);
  const [retrieving, setRetrieving] = useState(false);
  const [retrieveError, setRetrieveError] = useState('');
  const retrievalController = useRef<AbortController | null>(null);
  const downloadController = useRef<AbortController | null>(null);
  const contextController = useRef<AbortController | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState('');
  const [context, setContext] = useState<Citation | null>(null);
  const [contextParagraphs, setContextParagraphs] = useState<Paragraph[]>([]);
  const [contextLoading, setContextLoading] = useState(false);
  const [contextError, setContextError] = useState('');
  const [abstractExpanded, setAbstractExpanded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setSystemLoading(true); setSystemError(''); setJobsError('');
    Promise.allSettled([request<System>('/api/v1/system', controller.signal), request<{ items: Ingestion[] }>('/api/v1/ingestions', controller.signal)])
      .then(([status, ingestion]) => {
        if (controller.signal.aborted) return;
        if (status.status === 'fulfilled') setSystem(status.value);
        else { setSystem(null); setSystemError(errorMessage(status.reason)); }
        if (ingestion.status === 'fulfilled') setJobs(ingestion.value.items);
        else { setJobs([]); setJobsError(errorMessage(ingestion.reason)); }
      })
      .finally(() => { if (!controller.signal.aborted) setSystemLoading(false); });
    return () => controller.abort();
  }, [systemRevision]);

  useEffect(() => {
    const timer = setTimeout(() => { setFilter(search.trim()); setOffset(0); }, 250);
    return () => clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    const controller = new AbortController();
    setListLoading(true); setListError('');
    request<Page<Paper>>(`/api/v1/papers?q=${encodeURIComponent(filter)}&direction=${direction}&sort=${sort}&limit=${PAGE_SIZE}&offset=${offset}`, controller.signal)
      .then(page => { if (!controller.signal.aborted) { setPapers(page.items); setTotal(page.total); setSelected(current => current || page.items[0] || null); } })
      .catch(error => { if (!controller.signal.aborted && !isAbort(error)) setListError(errorMessage(error)); })
      .finally(() => { if (!controller.signal.aborted) setListLoading(false); });
    return () => controller.abort();
  }, [filter, direction, sort, offset, listRevision]);

  useEffect(() => () => { retrievalController.current?.abort(); downloadController.current?.abort(); contextController.current?.abort(); }, []);

  useEffect(() => {
    try { window.localStorage.setItem(LIBRARY_WIDTH_KEY, String(libraryWidth)); } catch { /* storage may be unavailable */ }
  }, [libraryWidth]);

  useEffect(() => {
    const fitToViewport = () => {
      if (window.innerWidth <= 600) return;
      const width = workbenchRef.current?.getBoundingClientRect().width;
      if (!width) return;
      const maximum = Math.max(MIN_LIBRARY_WIDTH,
        Math.min(MAX_LIBRARY_WIDTH, width - MIN_RESEARCH_WIDTH - RESIZE_HANDLE_WIDTH));
      setLibraryWidth(current => Math.round(Math.max(MIN_LIBRARY_WIDTH, Math.min(maximum, current))));
    };
    fitToViewport();
    window.addEventListener('resize', fitToViewport);
    return () => window.removeEventListener('resize', fitToViewport);
  }, []);

  function clampedLibraryWidth(clientX: number): number {
    const rect = workbenchRef.current?.getBoundingClientRect();
    if (!rect || !rect.width) return libraryWidth;
    const available = Math.max(MIN_LIBRARY_WIDTH, Math.min(MAX_LIBRARY_WIDTH,
      rect.width - MIN_RESEARCH_WIDTH - RESIZE_HANDLE_WIDTH));
    return Math.round(Math.max(MIN_LIBRARY_WIDTH, Math.min(available, clientX - rect.left)));
  }

  function startPaneResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (window.innerWidth <= 600) return;
    event.preventDefault();
    resizingRef.current = true; setResizingPane(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function movePaneResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (resizingRef.current) setLibraryWidth(clampedLibraryWidth(event.clientX));
  }

  function stopPaneResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (!resizingRef.current) return;
    resizingRef.current = false; setResizingPane(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function resizePaneWithKeyboard(event: ReactKeyboardEvent<HTMLDivElement>) {
    let next: number | null = null;
    if (event.key === 'ArrowLeft') next = libraryWidth - 16;
    if (event.key === 'ArrowRight') next = libraryWidth + 16;
    if (event.key === 'Home') next = MIN_LIBRARY_WIDTH;
    if (event.key === 'End') next = MAX_LIBRARY_WIDTH;
    if (next === null) return;
    event.preventDefault();
    const rect = workbenchRef.current?.getBoundingClientRect();
    const maximum = rect?.width ? Math.max(MIN_LIBRARY_WIDTH,
      Math.min(MAX_LIBRARY_WIDTH, rect.width - MIN_RESEARCH_WIDTH - RESIZE_HANDLE_WIDTH)) : MAX_LIBRARY_WIDTH;
    setLibraryWidth(Math.round(Math.max(MIN_LIBRARY_WIDTH, Math.min(maximum, next))));
  }

  function clearResult() { setResult(null); setRun(null); setRetrieveError(''); }

  function selectPaper(paper: Paper) {
    if (selected?.paper_id === paper.paper_id) return;
    retrievalController.current?.abort(); downloadController.current?.abort(); contextController.current?.abort();
    setSelected(paper); setQuestion(''); clearResult(); setRetrieving(false);
    setDownloading(false); setDownloadError(''); setContext(null); setAbstractExpanded(false);
  }

  function changeSort(value: PaperSort) {
    if (value === sort) return;
    retrievalController.current?.abort(); downloadController.current?.abort(); contextController.current?.abort();
    setSort(value); setOffset(0); setSelected(null); setQuestion(''); clearResult();
    setRetrieving(false); setDownloading(false); setDownloadError(''); setContext(null); setAbstractExpanded(false);
  }

  function changeDirection(value: ResearchDirection) {
    if (value === direction) return;
    retrievalController.current?.abort(); downloadController.current?.abort(); contextController.current?.abort();
    setDirection(value); setOffset(0); setSelected(null); setQuestion(''); clearResult();
    setRetrieving(false); setDownloading(false); setDownloadError(''); setContext(null); setAbstractExpanded(false);
  }

  async function retrieve(event?: FormEvent) {
    event?.preventDefault();
    if (!selected || !question.trim() || retrieving) return;
    retrievalController.current?.abort();
    const controller = new AbortController(); retrievalController.current = controller;
    setRetrieving(true); clearResult();
    try {
      const response = await request<Retrieval>(answerMode ? '/api/v1/answer' : '/api/v1/retrieve', controller.signal, { paper_id: selected.paper_id, query: question.trim(), top_k: topK, mode: retrievalMode, rerank, ...(retrievalMode === "hybrid" ? { rrf_constant: rrfConstant, dense_weight: denseWeight } : {}), ...(answerMode ? { profile: answerProfile, language: answerLanguage } : {}) });
      if (!controller.signal.aborted) { setResult(response); setRun(response.run || null); }
    } catch (error) {
      if (!controller.signal.aborted && !isAbort(error)) {
        setRetrieveError(errorMessage(error));
        setRun(error instanceof ApiError ? error.run || null : null);
      }
    }
    finally { if (!controller.signal.aborted) setRetrieving(false); }
  }

  function cancelRetrieve() { retrievalController.current?.abort(); setRetrieving(false); }

  async function download() {
    if (!selected || downloading) return;
    const controller = new AbortController(); downloadController.current = controller;
    setDownloading(true); setDownloadError('');
    try { await downloadSource(selected.paper_id, controller.signal); }
    catch (error) { if (!controller.signal.aborted && !isAbort(error)) setDownloadError(errorMessage(error)); }
    finally { if (!controller.signal.aborted) setDownloading(false); }
  }

  async function openContext(citation: Citation) {
    contextController.current?.abort();
    const controller = new AbortController(); contextController.current = controller;
    setContext(citation); setContextParagraphs([]); setContextLoading(true); setContextError('');
    try { const paragraphs = await loadContext(citation.paper_id, citation.chunk_id, controller.signal); if (!controller.signal.aborted) setContextParagraphs(paragraphs); }
    catch (error) { if (!controller.signal.aborted && !isAbort(error)) setContextError(errorMessage(error)); }
    finally { if (!controller.signal.aborted) setContextLoading(false); }
  }

  const connected = system?.database.status === 'ready' && system?.object_store.status === 'ready';
  return <div className="app-shell">
    <header className="app-header"><a className="brand" href="#" onClick={event => { event.preventDefault(); setTab('research'); }} aria-label="科研论文助手，返回论文工作台"><span className="brand-symbol"><Icon name="book" width="24" height="24" /></span><span><strong>研知<span className="brand-divider"> / </span></strong><span className="brand-caption">科研论文助手</span></span></a>
      <nav className="main-nav" aria-label="主导航"><button aria-current={tab === 'research' ? 'page' : undefined} className={tab === 'research' ? 'active' : ''} onClick={() => setTab('research')}><Icon name="book" />论文工作台</button><button aria-current={tab === 'storage' ? 'page' : undefined} className={tab === 'storage' ? 'active' : ''} onClick={() => setTab('storage')}><Icon name="database" />数据与存储</button></nav>
      <button className={`connection-state ${connected ? 'connected' : ''}`} onClick={() => setTab('storage')}><i /><span>{systemLoading ? '连接中' : systemError ? '服务连接异常' : connected ? '存储已连接' : system?.mode === 'files' ? '本地文件模式' : '存储待检查'}</span><Icon name="chevron" width="12" height="12" /></button>
    </header>
    {tab === 'storage' ? <StorageView system={system} jobs={jobs} loading={systemLoading} error={systemError} jobsError={jobsError} refresh={() => setSystemRevision(value => value + 1)} /> : <div ref={workbenchRef} className={`workbench ${resizingPane ? 'resizing' : ''}`} style={{ '--library-width': `${libraryWidth}px` } as CSSProperties}>
      <aside className="paper-library" aria-label="论文库">
        <div className="library-heading"><div><p className="eyebrow">YOUR RESEARCH LIBRARY</p><h1>论文库 <span>{number(system?.corpus.papers)}</span></h1></div><span className="source-mark">QASPER</span></div>
        <div className="search-field"><Icon name="search" /><input aria-label="搜索论文标题" value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索论文标题…" maxLength={200} />{search && <button className="icon-button" aria-label="清空搜索" onClick={() => setSearch('')}><Icon name="close" width="14" height="14" /></button>}</div>
        <div className="library-meta"><span>{filter ? `搜索结果 · ${number(total)} 篇` : direction === 'all' ? '全部论文' : `${DIRECTION_LABELS[direction]} · ${number(total)} 篇`}</span></div>
        <div className="library-controls"><label><span>研究方向</span><select aria-label="论文研究方向" value={direction} onChange={event => changeDirection(event.target.value as ResearchDirection)}>{Object.entries(DIRECTION_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label><span>方向内排序</span><select aria-label="论文排序方式" value={sort} onChange={event => changeSort(event.target.value as PaperSort)}>{Object.entries(SORT_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label></div>
        <div className="paper-list" aria-busy={listLoading}>
          {listError ? <ErrorNotice message={listError} retry={() => setListRevision(value => value + 1)} /> : listLoading ? <div className="paper-skeletons" role="status" aria-label="正在读取论文">{[1, 2, 3, 4, 5].map(item => <div className="paper-skeleton" key={item}><i /><i /><i /></div>)}</div> : papers.length ? papers.map(paper => <button className={`paper-item ${selected?.paper_id === paper.paper_id ? 'selected' : ''}`} key={paper.paper_id} onClick={() => selectPaper(paper)} aria-pressed={selected?.paper_id === paper.paper_id}><div className="paper-item-top"><span className="paper-number mono">{paper.paper_id}</span>{paper.ccf_level ? <span className={`ccf-badge ccf-${paper.ccf_level.toLowerCase()}`} title="CCF 对论文发表载体的目录级别；不代表论文质量评分">{`CCF ${paper.ccf_level}`}</span> : <Icon name="file" width="15" height="15" />}</div><h2 lang="en">{paper.title}</h2><div className="paper-item-foot"><span>{paper.research_direction_label || '其他方向'} · {paperDate(paper.arxiv_submitted_at)} · {paper.ccf_venue || '未分级'}</span><Icon name="arrow" width="16" height="16" /></div></button>) : <div className="library-empty"><Icon name="search" /><h2>没有找到论文</h2><p>{filter ? '换一个标题关键词试试。' : direction !== 'all' ? '这个方向暂时没有论文。' : '导入论文后，即可开始检索。'}</p></div>}
        </div>
        <div className="pagination"><span>{total ? `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)} / ${number(total)}` : '0 篇论文'}</span><div><button className="icon-button previous" aria-label="上一页论文" disabled={offset === 0 || listLoading} onClick={() => setOffset(value => Math.max(0, value - PAGE_SIZE))}><Icon name="chevron" /></button><button className="icon-button" aria-label="下一页论文" disabled={offset + PAGE_SIZE >= total || listLoading} onClick={() => setOffset(value => value + PAGE_SIZE)}><Icon name="chevron" /></button></div></div>
        <div className="library-note"><Icon name="info" /><p>方向取自 arXiv 主分类。先选方向，再在该方向内排序；CCF 未分级不代表低级别。</p></div>
      </aside>
      <div className="pane-resizer" role="separator" aria-label="调整论文库与正文区宽度" aria-orientation="vertical" aria-valuemin={MIN_LIBRARY_WIDTH} aria-valuemax={MAX_LIBRARY_WIDTH} aria-valuenow={libraryWidth} tabIndex={0} title="拖动调整左右占比；双击恢复默认宽度" onPointerDown={startPaneResize} onPointerMove={movePaneResize} onPointerUp={stopPaneResize} onPointerCancel={stopPaneResize} onLostPointerCapture={event => { if (resizingRef.current) stopPaneResize(event); }} onKeyDown={resizePaneWithKeyboard} onDoubleClick={() => setLibraryWidth(window.innerWidth <= 1100 ? 285 : 326)}><span /></div>
      <main className="research-main">
        <div className="research-breadcrumb"><span>论文工作台</span><Icon name="chevron" width="12" height="12" /><span>原文证据检索</span><span className="mode-tag">P2B · 证据问答</span></div>
        {selected ? <>
          <section className="paper-overview" aria-labelledby="paper-heading"><div className="paper-overline"><span className="tag green">当前论文</span><span className="mono">{selected.paper_id}</span>{selected.research_direction_label && <span className="direction-badge" title={`arXiv 主分类 ${selected.arxiv_primary_category}`}>{selected.research_direction_label} · {selected.arxiv_primary_category}</span>}<span>{paperDate(selected.arxiv_submitted_at)} 首次提交 arXiv</span>{selected.ccf_level && <span className={`ccf-badge ccf-${selected.ccf_level.toLowerCase()}`} title="CCF 对发表载体的目录级别；不代表论文质量评分">{selected.ccf_venue} · CCF {selected.ccf_level}</span>}<span className="paper-version">{selected.version}</span></div><h1 id="paper-heading" lang="en">{selected.title}</h1>
            <p className={`paper-abstract ${abstractExpanded ? 'expanded' : ''}`} lang="en">{selected.abstract || '这篇论文没有提供摘要。'}</p>
            <div className="paper-actions"><button className="text-button" onClick={() => setAbstractExpanded(value => !value)} aria-expanded={abstractExpanded}>{abstractExpanded ? '收起摘要' : '展开摘要'}<Icon name="chevron" className={abstractExpanded ? 'rotate-up' : 'rotate-down'} width="13" height="13" /></button>{selected.arxiv_pdf_url ? <a className="text-button" href={selected.arxiv_pdf_url} target="_blank" rel="noreferrer" title="从 arXiv 打开原论文 PDF，可在 PDF 阅读器中下载"><Icon name="external" width="14" height="14" />打开/下载原论文 PDF</a> : <button className="text-button" disabled title="当前记录没有可核验的原论文 PDF 链接"><Icon name="external" width="14" height="14" />原论文 PDF 暂无</button>}<button className="text-button" onClick={download} disabled={downloading || system?.object_store.status !== 'ready'} title={system?.object_store.status !== 'ready' ? '对象存储连接后可下载' : '下载已校验的 RAG 结构化正文 JSON'}><Icon name="download" width="15" height="15" />{downloading ? '正在下载…' : '下载 RAG 结构化正文'}</button></div>
            {downloadError && <ErrorNotice message={downloadError} retry={download} />}
          </section>
          <section className="question-section" aria-labelledby="question-heading"><div className="question-heading"><div><span className="section-number">01</span><h2 id="question-heading">向这篇论文提问</h2></div><span>检索范围：当前论文</span></div>
            <form className="question-form" onSubmit={retrieve}><label htmlFor="research-question" className="sr-only">输入检索问题</label><textarea id="research-question" value={question} onChange={event => { setQuestion(event.target.value); clearResult(); }} placeholder="这篇论文使用了什么方法？输入英文关键词或问题，寻找原文证据…" maxLength={2000} rows={3} disabled={retrieving} />
              <div className="question-toolbar"><label htmlFor="answer-mode">任务 <select id="answer-mode" value={answerMode ? "answer" : "evidence"} disabled={retrieving} onChange={event => { setAnswerMode(event.target.value === "answer"); clearResult(); }}><option value="evidence">检索证据</option><option value="answer">生成并核验回答</option></select></label><label htmlFor="retrieval-mode">检索方式 <select id="retrieval-mode" value={retrievalMode} onChange={event => { setRetrievalMode(event.target.value); clearResult(); }} disabled={retrieving}><option value="bm25">BM25 词法</option><option value="dense">向量语义</option><option value="hybrid">混合 RRF</option></select></label><label htmlFor="top-k">返回证据 <select id="top-k" value={topK} onChange={event => { setTopK(Number(event.target.value)); clearResult(); }} disabled={retrieving}><option value={5}>Top 5</option><option value={10}>Top 10</option><option value={20}>Top 20</option></select></label><label className="rerank-toggle"><input type="checkbox" checked={rerank} disabled={retrieving} onChange={event => { setRerank(event.target.checked); clearResult(); }} />模型重排</label><div className="question-submit"><span className="character-count">{question.length}/2000</span>{retrieving ? <button type="button" className="primary-button" onClick={cancelRetrieve}><Icon name="close" />{answerMode ? '停止等待' : '取消检索'}</button> : <button type="submit" className="primary-button" disabled={!question.trim()}><Icon name="search" />{answerMode ? '生成并核验回答' : '检索证据'}<Icon name="arrow" width="16" height="16" /></button>}</div></div>
              {(retrievalMode === 'hybrid' || answerMode) && <details className="experiment-options"><summary>实验设置</summary><div className="experiment-fields">
                {retrievalMode === 'hybrid' && <><label htmlFor="rrf-constant">RRF 平滑常数 K <select id="rrf-constant" value={rrfConstant} disabled={retrieving} onChange={e => { setRrfConstant(Number(e.target.value)); clearResult(); }}>
                  {[10, 30, 60, 100].map(k => <option key={k} value={k}>{k}</option>)}</select></label>
                  <label htmlFor="dense-weight">向量融合权重 <select id="dense-weight" value={denseWeight} disabled={retrieving} onChange={e => { setDenseWeight(Number(e.target.value)); clearResult(); }}>
                  {[0, 0.25, 0.5, 0.75, 1].map(w => <option key={w} value={w}>{w * 100}%{w === 0.5 ? '（等权）' : ''}</option>)}</select></label></>}
                {answerMode && <><label htmlFor="answer-profile">回答策略 <select id="answer-profile" value={answerProfile} disabled={retrieving} onChange={e => { setAnswerProfile(e.target.value); clearResult(); }}><option value="v1">v1 基线</option><option value="v2">v2 逐条支持核验（实验）</option></select></label>
                  <label htmlFor="answer-language">回答语言 <select id="answer-language" value={answerLanguage} disabled={retrieving} onChange={e => { setAnswerLanguage(e.target.value); clearResult(); }}><option value="zh">中文</option><option value="en">英文</option></select></label></>}
              </div><p>K 控制排名贡献的平滑程度，不是返回段落数。向量权重之外的部分分配给词法检索。实验策略仍可能误判，需对照原文。</p></details>}
            </form>{answerMode && <p className="generation-hint">{system?.capabilities.generation ? `生成模型：${system.generation?.model || '已配置'} · 每题最多两次 API 调用；仅展示核验通过的结论。` : '尚未接入生成模型，提交后仅返回证据和配置提示。'} 停止等待不会保证远端停止生成或计费。</p>}<p className="query-hint"><Icon name="info" width="14" height="14" />支持词法、向量和混合检索对比。当前模型针对英文，建议输入英文问题；可选模型重排会增加等待时间；重排问题限 128 token，首次调用需要加载本地模型。</p>
          </section>
          {result?.mode === 'grounded_answer' && <section className={`answer-section ${result.status === 'answered' ? 'verified' : ''}`} aria-labelledby="answer-heading">
            <div className="answer-heading"><h2 id="answer-heading">{result.status === 'answered' ? '基于证据的回答' : result.status === 'evidence_insufficient' ? '本次证据不足' : result.status === 'verification_failed' ? '回答未通过核验' : result.status === 'not_configured' ? '生成模型尚未配置' : '本次未生成答案'}</h2><span className="tag">{result.status === 'answered' ? '模型核验通过' : '仅展示证据'}</span></div>
            <p className="answer-notice">{result.notice}</p>
            {result.status === 'answered' && result.claims?.map((claim, index) => <article className="answer-claim" key={index}><p>{claim.text}</p><div className="claim-references">{claim.evidence.map((ref, refIndex) => { const citation = result.citations.find(c => c.chunk_id === ref.chunk_id); return <details key={`${ref.chunk_id}-${refIndex}`}><summary>引用 {index + 1}.{refIndex + 1} · {citation?.section_name || ref.chunk_id}</summary><blockquote lang="en">{ref.quote}</blockquote>{citation && <button className="text-button" onClick={() => void openContext(citation)}>查看原文上下文<Icon name="external" width="13" height="13" /></button>}</details>; })}</div></article>)}
            <div className="answer-meta"><span>{result.generation?.prompt_version ? `${result.generation.prompt_version} · ${result.generation.answer_language === 'en' ? '英文' : '中文'} · ` : ''}{result.generation?.model_calls || 0} 次生成/核验调用 · {((result.generation?.latency_ms || 0) / 1000).toFixed(1)} 秒（含检索）</span><span className="mono">Trace {result.trace_id}</span></div>
          </section>}
          {run && !retrieving && <RunRecordView key={run.trace_id} run={run} />}
          <section className="evidence-section" aria-labelledby="evidence-heading"><div className="evidence-heading"><div><span className="section-number">02</span><h2 id="evidence-heading">原文证据</h2>{result && <span className="evidence-count">{result.citations.length}</span>}</div>{result && <span className="retrieval-time"><Icon name="clock" width="13" height="13" />{result.trace.latency_ms.toFixed(1)} ms</span>}</div>
            {retrieveError && <ErrorNotice message={retrieveError} retry={() => void retrieve()} />}
            {retrieving ? <div className="evidence-empty" role="status"><span className="empty-symbol"><Icon name="refresh" className="spin" width="26" height="26" /></span><h3>{answerMode ? '正在检索、生成并核验' : '正在寻找原文证据'}</h3><p>{answerMode ? '核验完成后才会显示回答，请稍候…' : '在当前论文中检索与问题匹配的段落…'}</p></div> : result ? <><div className="result-query"><span>本次检索 · {result.trace.retriever === 'hybrid' ? '混合 RRF' : result.trace.retriever === 'dense' ? '向量语义' : 'BM25 词法'}{result.trace.rerank_enabled ? ' + 模型重排' : ''}{result.trace.retriever === 'hybrid' && result.trace.rrf_constant !== undefined ? ` · K=${result.trace.rrf_constant} · 向量权重 ${Math.round((result.trace.dense_weight ?? 0.5) * 100)}%` : ''}</span><p>{result.query}</p></div>{result.citations.length ? <div className="evidence-list">{result.citations.map(citation => <article className="evidence-card" key={citation.chunk_id}><div className="evidence-card-heading"><span className="evidence-rank">{String(citation.rank).padStart(2, '0')}</span><h3>{citation.section_name || '未命名章节'}</h3><span className="score-label" title="排序分数，不代表答案置信度；不同检索方式的分数不能直接比较">{result.trace.rerank_enabled ? '重排' : result.trace.retriever === 'hybrid' ? 'RRF' : result.trace.retriever === 'dense' ? '余弦' : 'BM25'} <strong>{citation.score.toFixed(!result.trace.rerank_enabled && result.trace.retriever === 'hybrid' ? 4 : 2)}</strong></span></div><p className="evidence-text" lang="en">{citation.text}</p><div className="evidence-card-foot"><span className="mono" title={citation.chunk_id}>{citation.chunk_id}</span><button className="text-button" onClick={() => void openContext(citation)}>查看原文上下文<Icon name="external" width="13" height="13" /></button></div></article>)}</div> : <div className="evidence-empty"><span className="empty-symbol"><Icon name="search" width="26" height="26" /></span><h3>没有找到词项匹配的段落</h3><p>试试论文中的方法名、数据集名或英文关键词。<br />未命中不代表这篇论文无法回答。</p></div>}<div className="result-footer"><p><Icon name="info" width="14" height="14" />{result.notice}</p><span className="mono">Trace {result.trace_id}</span></div></> : !retrieveError && <div className="evidence-empty"><span className="empty-symbol"><Icon name="file" width="26" height="26" /></span><h3>让答案的线索先出现</h3><p>输入一个问题，检索这篇论文中的相关段落。<br />每条结果都能回到原文，逐一核对。</p><span className="empty-caption">原文引用 · 相关性排序 · 上下文核对</span></div>}
          </section>
        </> : <div className="evidence-empty initial-empty"><span className="empty-symbol"><Icon name="book" width="30" height="30" /></span><h2>{listLoading ? '正在打开论文库' : '选择一篇论文，开始探索'}</h2><p>从左侧选择论文，查看摘要并检索原文。</p></div>}
        <footer className="workbench-footer"><span>研知 · Research with evidence</span><span>证据可追溯 · 模型核验不等于绝对正确</span></footer>
      </main>
    </div>}
    {context && <ContextDialog citation={context} paragraphs={contextParagraphs} loading={contextLoading} error={contextError} close={() => { contextController.current?.abort(); setContext(null); }} retry={() => void openContext(context)} />}
  </div>;
}
