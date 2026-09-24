import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import App from './App';
import { RESUME_KEY } from './conversationResume';

const papers = ['p1', 'p2'].map(paper_id => ({ paper_id, title: `Paper ${paper_id}`, abstract: '', source: 'qasper', version: 'v1' }));
const initial = { conversation_id: 'a'.repeat(32), paper_id: 'p1', paper_version: 'v1', revision: 0, turns: [], expires_at: '2026-09-25T12:00:00Z' };
const turn = { run_id: 'b'.repeat(32), question: 'alpha method', answer: 'Uses alpha evidence.', claim_count: 1, answer_truncated: false };
const saved = { ...initial, revision: 1, turns: [turn] };
const result = { trace_id: turn.run_id, query: 'alpha method', paper_id: 'p1', status: 'answered', mode: 'grounded_answer', claims: [{ text: turn.answer, evidence: [] }], citations: [], notice: '回答已核验', trace: { latency_ms: 1 } };
const json = (body: unknown) => new Response(JSON.stringify(body));

function setup(answer?: (options: RequestInit) => Promise<Response>, clear?: () => Promise<Response>) {
  const fetch = vi.fn((url: string, options: RequestInit) => {
    if (url === '/api/v1/system') return Promise.resolve(json({ mode: 'files', database: {}, object_store: {}, corpus: {}, access: { mode: 'local_public', principal_id: 'local-user' }, capabilities: { generation: true } }));
    if (url === '/api/v1/ingestions' || url.startsWith('/api/v1/runs')) return Promise.resolve(json({ items: [] }));
    if (url.startsWith('/api/v1/papers?')) return Promise.resolve(json({ total: 2, items: papers }));
    if (url === '/api/v1/papers/p1') return Promise.resolve(json(papers[0]));
    if (url === '/api/v1/conversations') return Promise.resolve(json(initial));
    if (url.endsWith('/clear')) return clear ? clear() : Promise.resolve(json({ cleared: true }));
    if (url.startsWith('/api/v1/conversations/')) return Promise.resolve(json(saved));
    if (url === '/api/v1/answer') return answer ? answer(options) : Promise.resolve(json({ ...result, conversation: saved }));
    throw new Error(url);
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

async function open() {
  render(<App />);
  await screen.findByRole('heading', { level: 1, name: 'Paper p1' });
  fireEvent.change(screen.getByLabelText('任务'), { target: { value: 'answer' } });
}

function submit(query = 'alpha method') {
  fireEvent.change(screen.getByLabelText('输入检索问题'), { target: { value: query } });
  fireEvent.click(screen.getByRole('button', { name: '生成并核验回答' }));
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); window.localStorage.clear(); window.sessionStorage.clear(); });

it('opts in, sends revision on follow-up, refreshes and clears on the server', async () => {
  const fetch = setup(); await open();
  expect((screen.getByRole('checkbox', { name: '记住本篇对话' }) as HTMLInputElement).checked).toBe(false);
  fireEvent.click(screen.getByRole('checkbox', { name: '记住本篇对话' }));
  submit();
  await screen.findByText(/已保留 1 轮/);
  submit('What are its limitations?');
  await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url === '/api/v1/answer')).toHaveLength(2));
  const sent = fetch.mock.calls.filter(([url]) => url === '/api/v1/answer').map(([, opts]) => JSON.parse(String(opts.body)));
  expect(sent.map(body => body.conversation_revision)).toEqual([0, 1]);
  expect(sent[1].conversation_id).toBe(initial.conversation_id);
  await waitFor(() => expect((screen.getByRole('button', { name: '清空对话' }) as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button', { name: '刷新对话' }));
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/v1/conversations/' + initial.conversation_id)).toBe(true));
  await waitFor(() => expect((screen.getByRole('button', { name: '清空对话' }) as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button', { name: '清空对话' }));
  await screen.findByText(/本篇对话记忆已清空/);
  expect(screen.queryByText(/已保留 1 轮/)).toBeNull();
  expect(sessionStorage.getItem(RESUME_KEY)).toBeNull();
});

it('does not create or send memory when disabled', async () => {
  const fetch = setup(() => Promise.resolve(json(result))); await open(); submit();
  await screen.findByText(turn.answer);
  expect(fetch.mock.calls.some(([url]) => url === '/api/v1/conversations')).toBe(false);
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => url === '/api/v1/answer')![1].body));
  expect(body.conversation_id).toBeUndefined();
});

it('ignores a late answer and its memory after switching papers', async () => {
  let resolve: (value: Response) => void = () => {};
  const fetch = setup(() => new Promise(r => { resolve = r; })); await open();
  fireEvent.click(screen.getByRole('checkbox', { name: '记住本篇对话' })); submit();
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => url === '/api/v1/answer')).toBe(true));
  fireEvent.click(screen.getByRole('button', { name: /Paper p2/ }));
  await act(async () => resolve(json({ ...result, conversation: saved })));
  expect(screen.queryByText(/已保留 1 轮/)).toBeNull();
  expect(screen.queryByText(turn.answer)).toBeNull();
  expect((screen.getByRole('checkbox', { name: '记住本篇对话' }) as HTMLInputElement).checked).toBe(false);
});

it('does not claim deletion when the server rejects clear', async () => {
  setup(undefined, () => Promise.resolve(new Response(JSON.stringify({ detail: '服务暂不可用' }), { status: 503 })));
  await open(); fireEvent.click(screen.getByRole('checkbox', { name: '记住本篇对话' })); submit();
  await screen.findByText(/已保留 1 轮/);
  fireEvent.click(screen.getByRole('button', { name: '清空对话' }));
  await screen.findByText('服务暂不可用');
  expect(screen.getByText(/已保留 1 轮/)).toBeTruthy();
  expect(screen.queryByText(/本篇对话记忆已清空/)).toBeNull();
});

it('reloads from server using only a stored pointer, without automatically calling the model', async () => {
  const fetch = setup(); await open();
  fireEvent.click(screen.getByRole('checkbox', { name: '记住本篇对话' })); submit();
  await screen.findByText(/已保留 1 轮/);
  const raw = sessionStorage.getItem(RESUME_KEY)!;
  expect(JSON.parse(raw)).toEqual({ conversation_id: initial.conversation_id, paper_id: 'p1', principal_id: 'local-user', mode: 'local_public', remember: true });
  expect(raw).not.toContain(turn.answer); expect(raw).not.toContain(turn.question);
  cleanup(); render(<App />);
  await screen.findByText('已恢复上次对话，可以继续追问。');
  expect(screen.getByText(/已保留 1 轮/)).toBeTruthy();
  expect(fetch.mock.calls.filter(([url]) => url === '/api/v1/answer')).toHaveLength(1);
  submit('What are its limitations?');
  await waitFor(() => expect(fetch.mock.calls.filter(([url]) => url === '/api/v1/answer')).toHaveLength(2));
  const body = JSON.parse(String(fetch.mock.calls.filter(([url]) => url === '/api/v1/answer')[1][1].body));
  expect(body.conversation_revision).toBe(1);
});

it('keeps paused memory paused after reload', async () => {
  setup(); await open(); fireEvent.click(screen.getByRole('checkbox', { name: '记住本篇对话' })); submit();
  await screen.findByText(/已保留 1 轮/);
  fireEvent.click(screen.getByRole('checkbox', { name: '记住本篇对话' }));
  cleanup(); render(<App />);
  await screen.findByText('已恢复上次对话；记忆仍暂停使用。');
  expect((screen.getByRole('checkbox', { name: '记住本篇对话' }) as HTMLInputElement).checked).toBe(false);
});
