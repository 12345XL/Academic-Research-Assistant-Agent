import { useEffect, useId, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiError, errorMessage, isAbort, request } from './api';
import type { AnswerFeedback, FeedbackTarget } from './api';

// Parent remounts this panel for a new run or identity. Never persist in browser storage.
export function AnswerFeedbackPanel({ runId, target }: { runId: string; target?: FeedbackTarget }) {
  const fieldId = useId();
  const controller = useRef<AbortController | null>(null);
  const [saved, setSaved] = useState<AnswerFeedback | null>(null);
  const [rating, setRating] = useState<AnswerFeedback['rating'] | ''>('');
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [conflict, setConflict] = useState(false);
  const [forbidden, setForbidden] = useState(false);
  const [message, setMessage] = useState('');
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const next = new AbortController(); controller.current = next;
    setLoading(true); setLoaded(false); setError(''); setMessage('');
    setConflict(false); setForbidden(false); setSaved(null); setRating(''); setNote('');
    request<{ feedback: AnswerFeedback | null }>(`/api/v1/runs/${encodeURIComponent(runId)}/feedback`, next.signal)
      .then(({ feedback }) => {
        if (next.signal.aborted) return;
        setSaved(feedback); setRating(feedback?.rating || ''); setNote(feedback?.note || ''); setLoaded(true);
      }).catch(failure => {
        if (next.signal.aborted || isAbort(failure)) return;
        setForbidden(failure instanceof ApiError && [401, 403, 404].includes(failure.status));
        setError(errorMessage(failure));
      }).finally(() => { if (!next.signal.aborted) setLoading(false); });
    return () => { next.abort(); controller.current?.abort(); };
  }, [runId, reload]);

  const artifact = saved?.target || target;
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!artifact || !rating || saving || loading || !loaded || conflict || forbidden) return;
    controller.current?.abort();
    const next = new AbortController(); controller.current = next;
    setSaving(true); setError(''); setMessage('');
    try {
      const response = await request<{ feedback: AnswerFeedback }>(`/api/v1/runs/${encodeURIComponent(runId)}/feedback`, next.signal,
        { target: artifact, rating, note, expected_revision: saved?.revision || 0 });
      if (next.signal.aborted) return;
      setSaved(response.feedback); setRating(response.feedback.rating); setNote(response.feedback.note);
      setMessage(`反馈已保存（版本 ${response.feedback.revision}）。不会自动修改答案或模型策略。`);
    } catch (failure) {
      if (next.signal.aborted || isAbort(failure)) return;
      const denied = failure instanceof ApiError && [401, 403, 404].includes(failure.status);
      setForbidden(denied);
      if (denied) { setSaved(null); setRating(''); setNote(''); }
      setConflict(failure instanceof ApiError && failure.status === 409);
      setError(failure instanceof ApiError && failure.status === 409
        ? '反馈版本已变化或回答快照不匹配。请重新读取后再修改，避免覆盖其他页面的反馈。'
        : `保存结果尚未确认。${errorMessage(failure)}`);
    } finally { if (!next.signal.aborted) setSaving(false); }
  }

  return <section className="answer-feedback" aria-label="回答反馈">
    <h3>这次回答对你有帮助吗？</h3>
    <p className="run-record-hint">提交后保存本次问题、已发布回答、引用定位及反馈；访问范围与运行记录一致。反馈是用户意见，不是正确性标注。</p>
    {loading && <p role="status">正在读取反馈…</p>}
    {error && <p role="alert">{error}</p>}
    {!forbidden && loaded && artifact && <>
      {!target && <details className="feedback-snapshot"><summary>查看这条反馈对应的问题与回答</summary>
        <p><strong>问题：</strong>{artifact.question}</p>
        {artifact.answer.claims.map((claim, index) => <div key={index}><p>{claim.text}</p>
          {claim.evidence.map((ref, i) => <blockquote key={i}>{ref.quote}<code>{ref.chunk_id}</code></blockquote>)}</div>)}
        <p className="run-record-hint">这是当时已发布回答的快照，不代表当前论文版本或已由人工确认正确。</p>
        {artifact.evidence.map(ref => <p className="mono" key={ref.chunk_id}>{ref.chunk_id} · 论文版本 {ref.version || '未记录'} · SHA-256 {ref.text_sha256}</p>)}
      </details>}
      <form onSubmit={save}>
        <fieldset disabled={saving || conflict}><legend className="sr-only">反馈评价</legend>
          <label><input type="radio" name={fieldId} checked={rating === 'helpful'} onChange={() => { setRating('helpful'); setMessage(''); }} />有帮助</label>
          <label><input type="radio" name={fieldId} checked={rating === 'problem'} onChange={() => { setRating('problem'); setMessage(''); }} />存在问题</label>
          <label className="feedback-note" htmlFor={`${fieldId}-note`}>补充说明（可选）</label>
          <textarea id={`${fieldId}-note`} value={note} maxLength={2000} rows={3} onChange={event => { setNote(event.target.value); setMessage(''); }} placeholder="例如：没有回答问题中的实验条件，或引用与结论不符。" />
        </fieldset>
        <button className="secondary-button" type="submit" disabled={!rating || saving || conflict}>{saving ? '正在保存反馈…' : saved ? '更新反馈' : '提交反馈'}</button>
        {saved && <span className="run-record-hint"> 已保存版本 {saved.revision} · 修改会更新这条反馈</span>}
      </form>
    </>}
    {!loading && loaded && !artifact && <p className="run-record-hint">这次运行尚未提交反馈，也没有保存答案全文。历史元数据无法还原待评价的回答。</p>}
    {!loading && !saving && (error || saved) && <button className="text-button" onClick={() => setReload(value => value + 1)}>重新读取已保存反馈（替换当前编辑）</button>}
    {message && <p role="status">{message}</p>}
  </section>;
}

export function HistoricalFeedback({ runId }: { runId: string }) {
  const [open, setOpen] = useState(false);
  return <details className="historical-feedback" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>查看或修改这次运行的反馈</summary>
    {open && <AnswerFeedbackPanel runId={runId} />}
  </details>;
}
