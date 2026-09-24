import { act, cleanup, render, renderHook, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { StoredRun } from './api';
import { useRunMonitor } from './useRunMonitor';
import { RunProgress } from './RunRecord';

const id = '0123456789abcdef0123456789abcdef';
function snapshot(stage = 'generate', revision = 2, state: StoredRun['state'] = 'running', reason: string | null = null): StoredRun {
  return { run_id: id, paper_id: 'p1', revision, state, reason, cancel_requested: false,
    created_at: '', updated_at: '', metadata: {}, snapshot: { run: {
      trace_id: id, state, reason, terminal_stage: stage, stages: [{ name: stage, status: state === 'running' ? 'running' : 'completed', latency_ms: null }],
    } } };
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const tick = async (ms: number) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); };
beforeEach(() => vi.useFakeTimers());
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it('tolerates acceptance 404, follows server stages and stops after terminal without posting', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(json({}, 404))
    .mockResolvedValueOnce(json(snapshot())).mockResolvedValueOnce(json(snapshot('verify', 3)))
    .mockResolvedValueOnce(json(snapshot('publish', 4, 'completed', 'published')));
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(() => useRunMonitor(id, false));
  await tick(300); expect(result.current.phase).toBe('pending');
  await tick(1000); expect(result.current.record?.snapshot.run?.terminal_stage).toBe('generate');
  await tick(1000); expect(result.current.record?.snapshot.run?.terminal_stage).toBe('verify');
  await tick(1000); expect(result.current.phase).toBe('terminal');
  await tick(20_000); expect(fetch).toHaveBeenCalledTimes(4);
  expect(fetch.mock.calls.every(([url, options]) => url === `/api/v1/runs/${id}` && !options.method)).toBe(true);
});

it('does not overlap requests and aborts an old identity request while ignoring its late result', async () => {
  let finish: (value: Response) => void = () => {};
  let signal: AbortSignal;
  const fetch = vi.fn((_url, options) => { signal = options.signal; return new Promise(resolve => { finish = resolve; }); });
  vi.stubGlobal('fetch', fetch);
  const { result, unmount } = renderHook(() => useRunMonitor(id, false));
  await tick(300); await tick(3000); expect(fetch).toHaveBeenCalledTimes(1);
  unmount(); expect(signal!.aborted).toBe(true);
  await act(async () => finish(json(snapshot('publish', 10, 'completed', 'published'))));
  await tick(10_000); expect(fetch).toHaveBeenCalledTimes(1); expect(result.current.record).toBeNull();
});

it('backs off on storage errors and recovers an interrupted run after restart', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(json(snapshot()))
    .mockResolvedValueOnce(json({ detail: '存储暂不可用' }, 503))
    .mockResolvedValueOnce(json(snapshot('generate', 3, 'interrupted', 'server_restarted')));
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(() => useRunMonitor(id, false));
  await tick(300); await tick(1000);
  expect(result.current.phase).toBe('retrying'); expect(result.current.record?.state).toBe('running');
  await tick(1999); expect(fetch).toHaveBeenCalledTimes(2);
  await tick(1); expect(result.current.phase).toBe('terminal');
  expect(result.current.record?.reason).toBe('server_restarted'); expect(result.current.error).toBe('');
});

it.each([401, 403])('clears the old snapshot and stops querying on access error %s', async status => {
  const fetch = vi.fn().mockResolvedValueOnce(json(snapshot())).mockResolvedValueOnce(json({}, status));
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(() => useRunMonitor(id, false));
  await tick(1300); expect(result.current.phase).toBe('unavailable'); expect(result.current.record).toBeNull();
  await tick(20_000); expect(fetch).toHaveBeenCalledTimes(2);
});

it('bounds repeated missing records and only resumes GETs on explicit status retry', async () => {
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(json({}, 404)));
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(() => useRunMonitor(id, false));
  await tick(10_300); expect(result.current.phase).toBe('unavailable');
  const count = fetch.mock.calls.length;
  await tick(10_000); expect(fetch).toHaveBeenCalledTimes(count);
  fetch.mockResolvedValue(json(snapshot()));
  act(() => result.current.retry()); await tick(300);
  expect(result.current.phase).toBe('watching'); expect(fetch).toHaveBeenCalledTimes(count + 1);
});

it('times out a stuck status query without declaring the answer failed', async () => {
  vi.stubGlobal('fetch', vi.fn((_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
  })));
  const { result } = renderHook(() => useRunMonitor(id, false));
  await tick(5300); expect(result.current.phase).toBe('retrying');
  expect(result.current.error).toContain('问答是否完成尚未确认');
});

it('stops an in-flight observation when the answer response supplies its final run', async () => {
  let signal: AbortSignal;
  const fetch = vi.fn((_url, options) => { signal = options.signal; return new Promise(() => {}); });
  vi.stubGlobal('fetch', fetch);
  const { rerender } = renderHook(({ finished }) => useRunMonitor(id, finished), { initialProps: { finished: false } });
  await tick(300); rerender({ finished: true }); expect(signal!.aborted).toBe(true);
  await tick(20_000); expect(fetch).toHaveBeenCalledTimes(1);
});

it('does not regress revisions or accept another run as this run', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(json(snapshot('verify', 5)))
    .mockResolvedValueOnce(json(snapshot('generate', 4)))
    .mockResolvedValueOnce(json({ ...snapshot(), run_id: 'another-run' }));
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(() => useRunMonitor(id, false));
  await tick(2300); expect(result.current.record?.revision).toBe(5);
  expect(result.current.phase).toBe('retrying');
});

it('shows stage updates and publication status without displaying any draft or cancelling answer delivery', async () => {
  const terminal = snapshot('publish', 4, 'completed', 'published');
  const fetch = vi.fn().mockResolvedValueOnce(json({ ...snapshot(), claims: [{ text: 'NEVER_RENDER_DRAFT' }] }))
    .mockResolvedValueOnce(json(terminal));
  vi.stubGlobal('fetch', fetch);
  render(<RunProgress runId={id} waiting />);
  await tick(300); expect(screen.getByRole('status').textContent).toBe('当前阶段：生成回答');
  expect(screen.queryByText('NEVER_RENDER_DRAFT')).toBeNull();
  await tick(1000); expect(screen.getByText(/正在接收回答；状态记录不会代替答案正文/)).toBeTruthy();
});

it('automatically confirms cancellation from the server instead of leaving the acceptance notice', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(json({ ...snapshot(), cancel_requested: true }))
    .mockResolvedValueOnce(json(snapshot('generate', 4, 'failed', 'cancelled'))));
  render(<RunProgress runId={id} notice="取消已接受，仍待确认" />);
  await tick(300); expect(screen.getByRole('status').textContent).toContain('正在确认终态');
  await tick(1000); expect(screen.getByRole('status').textContent).toBe('运行已取消');
  expect(screen.queryByText('取消已接受，仍待确认')).toBeNull();
});

it('resumes observation when an unfinished historical run is opened after page reload', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(snapshot('generate', 3, 'interrupted', 'server_restarted'))));
  render(<RunProgress runId={id} initial={snapshot()} />);
  expect(screen.getByRole('status').textContent).toBe('当前阶段：生成回答');
  await tick(300); expect(screen.getByRole('status').textContent).toContain('服务重启');
});
