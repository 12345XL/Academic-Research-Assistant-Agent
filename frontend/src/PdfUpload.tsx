import { useEffect, useId, useRef, useState } from 'react';
import { errorMessage, isAbort, uploadPdf } from './api';
import type { Paper, PdfUploadResult } from './api';

export function PdfUpload({ onOpen }: { onOpen: (paper: Paper) => void }) {
  const id = useId();
  const controller = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<PdfUploadResult | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function submit() {
    if (!file || busy) return;
    if (!file.size || file.size > 10 * 1024 * 1024) { setError('请选择非空且不超过 10 MiB 的 PDF。'); return; }
    const next = new AbortController(); controller.current = next;
    setBusy(true); setError(''); setResult(null);
    try {
      const saved = await uploadPdf(file, next.signal);
      if (!next.signal.aborted) setResult(saved);
    } catch (failure) {
      if (!next.signal.aborted && !isAbort(failure)) setError(`${errorMessage(failure)} 若连接中断，可重新上传同一文件确认结果，不会重复新增论文。`);
    } finally { if (!next.signal.aborted) setBusy(false); }
  }
  return <details className="pdf-upload"><summary>上传文本 PDF</summary>
    <p>本机公共模式：文件与提取正文会保存到本机配置的存储中，使用此工作台的人可见。上传不会调用回答模型。</p>
    <p>最多 10 MiB、80 页。支持可提取文字的 PDF；扫描件不做 OCR，双栏、表格和公式请对照原文件检查。</p>
    <label htmlFor={id}>选择 PDF 文件</label>
    <input id={id} type="file" accept="application/pdf,.pdf" disabled={busy} onChange={event => { setFile(event.target.files?.[0] || null); setResult(null); setError(''); }} />
    <button className="secondary-button" disabled={busy || !file} onClick={() => void submit()}>{busy ? '正在解析并入库…' : '上传并解析'}</button>
    {busy && <p role="status">正在处理。关闭页面不代表后台已取消；处理完成后才会出现在论文库。</p>}
    {error && <p role="alert">{error}</p>}
    {result && <div className="pdf-upload-result"><p role="status">{result.outcome === 'skipped' ? '已复用已有论文' : '已完成入库'}：{result.page_count} 页，{result.paragraphs} 个文本块。</p>
      {result.warnings.map((warning, index) => <p key={index}>{warning}</p>)}
      <details><summary>核对提取文字样例</summary>{result.preview.map(chunk => <div key={chunk.chunk_id}><strong>{chunk.section_name}</strong><p>{chunk.text}</p></div>)}</details>
      <button className="primary-button" onClick={() => onOpen(result.paper)}>打开论文并提问</button>
    </div>}
  </details>;
}
