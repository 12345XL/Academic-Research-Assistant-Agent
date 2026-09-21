import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import Icon from './Icon';
import { downloadSource, errorMessage, isAbort, loadContext, request } from './api';
import type { Citation, Ingestion, Page, Paper, Paragraph, Retrieval, System } from './api';

const PAGE_SIZE = 12;
const number = (value: number | undefined) => value === undefined ? '—' : value.toLocaleString('zh-CN');
const date = (value?: string | null) => value ? new Date(value.replace(' ', 'T')).toLocaleString('zh-CN', { hour12: false }) : '—';

function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="error-notice" role="alert"><Icon name="info" /><span>{message}</span>{retry && <button onClick={retry}>重试</button>}</div>;
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
  const [tab, setTab] = useState<'research' | 'storage'>('research');
  const [system, setSystem] = useState<System | null>(null);
  const [jobs, setJobs] = useState<Ingestion[]>([]);
  const [systemLoading, setSystemLoading] = useState(true);
  const [systemError, setSystemError] = useState('');
  const [jobsError, setJobsError] = useState('');
  const [systemRevision, setSystemRevision] = useState(0);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('');
  const [offset, setOffset] = useState(0);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [total, setTotal] = useState(0);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState('');
  const [listRevision, setListRevision] = useState(0);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<Retrieval | null>(null);
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
    request<Page<Paper>>(`/api/v1/papers?q=${encodeURIComponent(filter)}&limit=${PAGE_SIZE}&offset=${offset}`, controller.signal)
      .then(page => { if (!controller.signal.aborted) { setPapers(page.items); setTotal(page.total); setSelected(current => current || page.items[0] || null); } })
      .catch(error => { if (!controller.signal.aborted && !isAbort(error)) setListError(errorMessage(error)); })
      .finally(() => { if (!controller.signal.aborted) setListLoading(false); });
    return () => controller.abort();
  }, [filter, offset, listRevision]);

  useEffect(() => () => { retrievalController.current?.abort(); downloadController.current?.abort(); contextController.current?.abort(); }, []);

  function selectPaper(paper: Paper) {
    if (selected?.paper_id === paper.paper_id) return;
    retrievalController.current?.abort(); downloadController.current?.abort(); contextController.current?.abort();
    setSelected(paper); setQuestion(''); setResult(null); setRetrieving(false); setRetrieveError('');
    setDownloading(false); setDownloadError(''); setContext(null); setAbstractExpanded(false);
  }

  async function retrieve(event?: FormEvent) {
    event?.preventDefault();
    if (!selected || !question.trim() || retrieving) return;
    retrievalController.current?.abort();
    const controller = new AbortController(); retrievalController.current = controller;
    setRetrieving(true); setRetrieveError(''); setResult(null);
    try {
      const response = await request<Retrieval>('/api/v1/retrieve', controller.signal, { paper_id: selected.paper_id, query: question.trim(), top_k: topK });
      if (!controller.signal.aborted) setResult(response);
    } catch (error) { if (!controller.signal.aborted && !isAbort(error)) setRetrieveError(errorMessage(error)); }
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
    {tab === 'storage' ? <StorageView system={system} jobs={jobs} loading={systemLoading} error={systemError} jobsError={jobsError} refresh={() => setSystemRevision(value => value + 1)} /> : <div className="workbench">
      <aside className="paper-library" aria-label="论文库">
        <div className="library-heading"><div><p className="eyebrow">YOUR RESEARCH LIBRARY</p><h1>论文库 <span>{number(system?.corpus.papers)}</span></h1></div><span className="source-mark">QASPER</span></div>
        <div className="search-field"><Icon name="search" /><input aria-label="搜索论文标题" value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索论文标题…" maxLength={200} />{search && <button className="icon-button" aria-label="清空搜索" onClick={() => setSearch('')}><Icon name="close" width="14" height="14" /></button>}</div>
        <div className="library-meta"><span>{filter ? `搜索结果 · ${number(total)} 篇` : '全部论文'}</span><span>按论文编号排序</span></div>
        <div className="paper-list" aria-busy={listLoading}>
          {listError ? <ErrorNotice message={listError} retry={() => setListRevision(value => value + 1)} /> : listLoading ? <div className="paper-skeletons" role="status" aria-label="正在读取论文">{[1, 2, 3, 4, 5].map(item => <div className="paper-skeleton" key={item}><i /><i /><i /></div>)}</div> : papers.length ? papers.map(paper => <button className={`paper-item ${selected?.paper_id === paper.paper_id ? 'selected' : ''}`} key={paper.paper_id} onClick={() => selectPaper(paper)} aria-pressed={selected?.paper_id === paper.paper_id}><div className="paper-item-top"><span className="paper-number mono">{paper.paper_id}</span><Icon name="file" width="15" height="15" /></div><h2 lang="en">{paper.title}</h2><div className="paper-item-foot"><span>NLP · {paper.source.toUpperCase()}</span><Icon name="arrow" width="16" height="16" /></div></button>) : <div className="library-empty"><Icon name="search" /><h2>没有找到论文</h2><p>{filter ? '换一个标题关键词试试。' : '导入论文后，即可开始检索。'}</p></div>}
        </div>
        <div className="pagination"><span>{total ? `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)} / ${number(total)}` : '0 篇论文'}</span><div><button className="icon-button previous" aria-label="上一页论文" disabled={offset === 0 || listLoading} onClick={() => setOffset(value => Math.max(0, value - PAGE_SIZE))}><Icon name="chevron" /></button><button className="icon-button" aria-label="下一页论文" disabled={offset + PAGE_SIZE >= total || listLoading} onClick={() => setOffset(value => value + PAGE_SIZE)}><Icon name="chevron" /></button></div></div>
        <div className="library-note"><Icon name="layers" /><p>从论文出发，让每个发现<br />回到可核对的原文。</p></div>
      </aside>
      <main className="research-main">
        <div className="research-breadcrumb"><span>论文工作台</span><Icon name="chevron" width="12" height="12" /><span>原文证据检索</span><span className="mode-tag">BM25 · 当前阶段</span></div>
        {selected ? <>
          <section className="paper-overview" aria-labelledby="paper-heading"><div className="paper-overline"><span className="tag green">当前论文</span><span className="mono">{selected.paper_id}</span><span className="paper-version">{selected.version}</span></div><h1 id="paper-heading" lang="en">{selected.title}</h1>
            <p className={`paper-abstract ${abstractExpanded ? 'expanded' : ''}`} lang="en">{selected.abstract || '这篇论文没有提供摘要。'}</p>
            <div className="paper-actions"><button className="text-button" onClick={() => setAbstractExpanded(value => !value)} aria-expanded={abstractExpanded}>{abstractExpanded ? '收起摘要' : '展开摘要'}<Icon name="chevron" className={abstractExpanded ? 'rotate-up' : 'rotate-down'} width="13" height="13" /></button><button className="text-button" onClick={download} disabled={downloading || system?.object_store.status !== 'ready'} title={system?.object_store.status !== 'ready' ? '对象存储连接后可下载' : '下载已校验的结构化原文 JSON'}><Icon name="download" width="15" height="15" />{downloading ? '正在下载…' : '下载结构化原文'}</button></div>
            {downloadError && <ErrorNotice message={downloadError} retry={download} />}
          </section>
          <section className="question-section" aria-labelledby="question-heading"><div className="question-heading"><div><span className="section-number">01</span><h2 id="question-heading">向这篇论文提问</h2></div><span>检索范围：当前论文</span></div>
            <form className="question-form" onSubmit={retrieve}><label htmlFor="research-question" className="sr-only">输入检索问题</label><textarea id="research-question" value={question} onChange={event => setQuestion(event.target.value)} placeholder="这篇论文使用了什么方法？输入英文关键词或问题，寻找原文证据…" maxLength={2000} rows={3} disabled={retrieving} />
              <div className="question-toolbar"><label htmlFor="top-k">返回证据 <select id="top-k" value={topK} onChange={event => setTopK(Number(event.target.value))} disabled={retrieving}><option value={5}>Top 5</option><option value={10}>Top 10</option><option value={20}>Top 20</option></select></label><div className="question-submit"><span className="character-count">{question.length}/2000</span>{retrieving ? <button type="button" className="primary-button" onClick={cancelRetrieve}><Icon name="close" />取消检索</button> : <button type="submit" className="primary-button" disabled={!question.trim()}><Icon name="search" />检索证据<Icon name="arrow" width="16" height="16" /></button>}</div></div>
            </form><p className="query-hint"><Icon name="info" width="14" height="14" />当前使用英文词项匹配。英文问题与关键词更适合此语料，中文语义检索尚未接入。</p>
          </section>
          <section className="evidence-section" aria-labelledby="evidence-heading"><div className="evidence-heading"><div><span className="section-number">02</span><h2 id="evidence-heading">原文证据</h2>{result && <span className="evidence-count">{result.citations.length}</span>}</div>{result && <span className="retrieval-time"><Icon name="clock" width="13" height="13" />{result.trace.latency_ms.toFixed(1)} ms</span>}</div>
            {retrieveError && <ErrorNotice message={retrieveError} retry={() => void retrieve()} />}
            {retrieving ? <div className="evidence-empty" role="status"><span className="empty-symbol"><Icon name="refresh" className="spin" width="26" height="26" /></span><h3>正在寻找原文证据</h3><p>在当前论文中检索与问题匹配的段落…</p></div> : result ? <><div className="result-query"><span>本次检索</span><p>{result.query}</p></div>{result.citations.length ? <div className="evidence-list">{result.citations.map(citation => <article className="evidence-card" key={citation.chunk_id}><div className="evidence-card-heading"><span className="evidence-rank">{String(citation.rank).padStart(2, '0')}</span><h3>{citation.section_name || '未命名章节'}</h3><span className="score-label" title="BM25 相关性分数，不代表置信度或概率">BM25 <strong>{citation.score.toFixed(2)}</strong></span></div><p className="evidence-text" lang="en">{citation.text}</p><div className="evidence-card-foot"><span className="mono" title={citation.chunk_id}>{citation.chunk_id}</span><button className="text-button" onClick={() => void openContext(citation)}>查看原文上下文<Icon name="external" width="13" height="13" /></button></div></article>)}</div> : <div className="evidence-empty"><span className="empty-symbol"><Icon name="search" width="26" height="26" /></span><h3>没有找到词项匹配的段落</h3><p>试试论文中的方法名、数据集名或英文关键词。<br />未命中不代表这篇论文无法回答。</p></div>}<div className="result-footer"><p><Icon name="info" width="14" height="14" />{result.notice}</p><span className="mono">Trace {result.trace_id}</span></div></> : !retrieveError && <div className="evidence-empty"><span className="empty-symbol"><Icon name="file" width="26" height="26" /></span><h3>让答案的线索先出现</h3><p>输入一个问题，检索这篇论文中的相关段落。<br />每条结果都能回到原文，逐一核对。</p><span className="empty-caption">原文引用 · 相关性排序 · 上下文核对</span></div>}
          </section>
        </> : <div className="evidence-empty initial-empty"><span className="empty-symbol"><Icon name="book" width="30" height="30" /></span><h2>{listLoading ? '正在打开论文库' : '选择一篇论文，开始探索'}</h2><p>从左侧选择论文，查看摘要并检索原文。</p></div>}
        <footer className="workbench-footer"><span>研知 · Research with evidence</span><span>当前仅检索原文，尚未生成或核验答案</span></footer>
      </main>
    </div>}
    {context && <ContextDialog citation={context} paragraphs={contextParagraphs} loading={contextLoading} error={contextError} close={() => { contextController.current?.abort(); setContext(null); }} retry={() => void openContext(context)} />}
  </div>;
}
