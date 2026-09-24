import { useEffect, useState } from 'react';
import { ApiError, errorMessage, request } from './api';
import type { StoredRun } from './api';

export type MonitorPhase = 'pending' | 'watching' | 'retrying' | 'terminal' | 'unavailable';
const MAX_WATCH_MS = 11 * 60_000;

// Read-only observation: never resubmit an answer or infer execution failure.
export function useRunMonitor(runId: string, finished: boolean, initial?: StoredRun) {
  const [record, setRecord] = useState<StoredRun | null>(initial || null);
  const [phase, setPhase] = useState<MonitorPhase>(initial?.state === 'running' ? 'watching' : 'pending');
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (finished || (initial && initial.state !== 'running')) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let failures = 0;
    let seen = Boolean(initial);
    let revision = initial?.revision || 0;
    const started = performance.now();
    setRecord(initial || null); setError(''); setPhase(initial ? 'watching' : 'pending');

    async function poll() {
      if (disposed) return;
      if (performance.now() - started >= MAX_WATCH_MS) {
        setPhase('unavailable');
        setError('自动查询已暂停，尚未确认运行终态。可重新查询，不会重新生成答案。');
        return;
      }
      controller = new AbortController();
      const current = controller;
      let timedOut = false;
      timeout = setTimeout(() => { timedOut = true; current.abort(); }, 5000);
      let delay = 1000;
      try {
        const next = await request<StoredRun>(`/api/v1/runs/${encodeURIComponent(runId)}`, current.signal);
        if (disposed || current.signal.aborted) return;
        if (next.run_id !== runId) throw new Error('收到不匹配的运行记录，暂未更新状态。');
        seen = true; failures = 0; setError('');
        if (next.revision >= revision) {
          revision = next.revision;
          setRecord(next);
          if (next.state !== 'running') { setPhase('terminal'); return; }
        }
        setPhase('watching');
      } catch (failure) {
        if (disposed) return;
        if (failure instanceof ApiError && [401, 403].includes(failure.status)) {
          setRecord(null); setPhase('unavailable');
          setError('当前凭证无法读取运行状态，已停止查询；这不表示后台已取消。');
          return;
        }
        if (failure instanceof ApiError && failure.status === 404) {
          // The first GET can beat answer acceptance/creation. This is not failure.
          if (!seen && performance.now() - started < 10_000) {
            setPhase('pending');
          } else {
            setRecord(null); setPhase('unavailable');
            setError('未找到可访问的运行记录，无法确认执行结果；没有重新提交问答。');
            return;
          }
        } else {
          failures++;
          delay = Math.min(8000, 1000 * 2 ** Math.min(failures, 3));
          setPhase('retrying');
          setError(timedOut ? '运行状态查询超时，将自动重试；问答是否完成尚未确认。'
            : `运行状态暂不可用，将自动重试。${errorMessage(failure)}`);
        }
      } finally { clearTimeout(timeout); }
      if (!disposed) timer = setTimeout(() => void poll(), delay);
    }

    timer = setTimeout(() => void poll(), 300);
    return () => { disposed = true; clearTimeout(timer); clearTimeout(timeout); controller?.abort(); };
  }, [runId, finished, initial, refresh]);

  return { record, phase, error, retry: () => setRefresh(value => value + 1) };
}
