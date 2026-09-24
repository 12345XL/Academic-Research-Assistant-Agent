import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import App from './App';
import { StrictMode } from 'react';
import { RESUME_KEY } from './conversationResume';

const pointer = { conversation_id: 'a'.repeat(32), paper_id: 'p2', principal_id: 'alice', mode: 'bearer_policy', remember: true };
const papers = ['p1', 'p2'].map(paper_id => ({ paper_id, title: `Paper ${paper_id}`, abstract: '', source: 'qasper', version: 'v1' }));
const conversation = { conversation_id: pointer.conversation_id, paper_id: 'p2', paper_version: 'v1', revision: 9,
  turns: [{ run_id: 'r1', question: 'Private question', answer: 'Private answer', claim_count: 1, answer_truncated: false }], expires_at: '2026-09-25T00:00:00Z' };
const system = { mode: 'postgres', database: {}, object_store: {}, corpus: {}, access: { mode: 'bearer_policy', principal_id: 'alice' }, capabilities: { generation: true } };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

function setup(read: () => Promise<Response>, status = () => Promise.resolve(json(system))) {
  sessionStorage.setItem(RESUME_KEY, JSON.stringify(pointer));
  const fetch = vi.fn((url: string, options: RequestInit) => {
    if (url === '/api/v1/system') return status();
    if (url === '/api/v1/ingestions' || url.startsWith('/api/v1/runs')) return Promise.resolve(json({ items: [] }));
    if (url.startsWith('/api/v1/papers?')) return Promise.resolve(json({ total: 2, items: papers }));
    if (url === '/api/v1/papers/p2') return Promise.resolve(json(papers[1]));
    if (url === '/api/v1/conversations/' + pointer.conversation_id) return read();
    if (url === '/api/v1/answer') return Promise.resolve(json({ query: 'followup', citations: [], trace: { latency_ms: 0 } }));
    throw new Error(url + String(options));
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

afterEach(() => { cleanup(); sessionStorage.clear(); localStorage.clear(); vi.unstubAllGlobals(); });

it('restores under StrictMode without submitting an answer', async () => {
  const fetch = setup(() => Promise.resolve(json(conversation)));
  render(<StrictMode><App /></StrictMode>);
  await screen.findByText('已恢复上次对话，可以继续追问。');
  expect(screen.getByText(/已保留 1 轮/)).toBeTruthy();
  expect(fetch.mock.calls.some(([url]) => url === '/api/v1/answer')).toBe(false);
});

it('ignores a malformed browser pointer', async () => {
  const fetch = setup(() => Promise.resolve(json(conversation)));
  sessionStorage.setItem(RESUME_KEY, '{not-json');
  render(<App />);
  await screen.findByRole('heading', { name: 'Paper p1', level: 1 });
  expect(fetch.mock.calls.some(([url]) => url.startsWith('/api/v1/conversations'))).toBe(false);
  expect(screen.queryByRole('button', { name: '重试恢复对话' })).toBeNull();
  expect(sessionStorage.getItem(RESUME_KEY)).toBeNull();
});

it.each([403, 404, 409])('drops revoked/expired/version-stale pointer on %s without displaying cached content', async code => {
  const fetch = setup(() => Promise.resolve(json({ detail: '旧会话不可用' }, code)));
  render(<App />);
  await screen.findByText(/旧对话未恢复/);
  expect(sessionStorage.getItem(RESUME_KEY)).toBeNull();
  expect(screen.queryByText('Private answer')).toBeNull();
  expect(fetch.mock.calls.some(([url]) => url === '/api/v1/answer')).toBe(false);
});

it('retries transient recovery errors and restores the selected paper outside the default selection', async () => {
  let attempts = 0;
  const fetch = setup(() => Promise.resolve(++attempts === 1 ? json({ detail: '临时故障' }, 503) : json(conversation)));
  render(<App />);
  await screen.findByText(/对话恢复暂未完成/);
  expect(sessionStorage.getItem(RESUME_KEY)).not.toBeNull();
  fireEvent.click(screen.getByRole('button', { name: '重试恢复对话' }));
  await screen.findByText('已恢复上次对话，可以继续追问。');
  expect(screen.getByRole('heading', { level: 1, name: 'Paper p2' })).toBeTruthy();
  fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: 'What are its limitations?' } });
  fireEvent.click(screen.getByRole('button', { name: '生成并核验回答' }));
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/v1/answer')).toBe(true));
  const submitted = JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/v1/answer')![1].body));
  expect(submitted.paper_id).toBe('p2'); expect(submitted.conversation_revision).toBe(9);
});

it('discards another identity before fetching the old conversation', async () => {
  const fetch = setup(() => Promise.resolve(json(conversation)), () => Promise.resolve(json({ ...system, access: { mode: 'bearer_policy', principal_id: 'bob' } })));
  render(<App />);
  await screen.findByText(/当前身份与上次对话不同/);
  expect(fetch.mock.calls.some(([url]) => url.startsWith('/api/v1/conversations'))).toBe(false);
  expect(sessionStorage.getItem(RESUME_KEY)).toBeNull();
});

it('waits for authentication after reload without storing credentials', async () => {
  let authenticated = false;
  const fetch = setup(() => Promise.resolve(json(conversation)), () => Promise.resolve(authenticated ? json(system) : json({ detail: '请重新输入凭证' }, 401)));
  render(<App />);
  await screen.findByText(/等待身份确认后恢复/);
  await screen.findByText('服务连接异常');
  expect(fetch.mock.calls.some(([url]) => url.startsWith('/api/v1/conversations'))).toBe(false);
  authenticated = true;
  fireEvent.change(screen.getByLabelText('Bearer 凭证'), { target: { value: 'test-only-token' } });
  fireEvent.click(screen.getByRole('button', { name: '应用凭证并重载' }));
  await screen.findByText('已恢复上次对话，可以继续追问。');
  expect(sessionStorage.getItem(RESUME_KEY)).not.toContain('test-only-token');
});

it.each(['paper', 'discard', 'identity'])('ignores a late restore after %s change', async action => {
  let resolve: (value: Response) => void = () => {};
  let principal = 'alice';
  const fetch = setup(() => new Promise(r => { resolve = r; }), () => Promise.resolve(json({ ...system, access: { mode: 'bearer_policy', principal_id: principal } })));
  render(<App />);
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.startsWith('/api/v1/conversations'))).toBe(true));
  if (action === 'paper') fireEvent.click(screen.getByRole('button', { name: /Paper p2/ }));
  if (action === 'discard') fireEvent.click(screen.getByRole('button', { name: '放弃恢复，开始新对话' }));
  if (action === 'identity') {
    principal = 'bob';
    fireEvent.change(screen.getByLabelText('Bearer 凭证'), { target: { value: 'new-token' } });
    fireEvent.click(screen.getByRole('button', { name: '应用凭证并重载' }));
  }
  await act(async () => resolve(json(conversation)));
  expect(screen.queryByText('已恢复上次对话，可以继续追问。')).toBeNull();
  await waitFor(() => expect(sessionStorage.getItem(RESUME_KEY)).toBeNull());
  expect(screen.queryByText(/Private answer/)).toBeNull();
});
