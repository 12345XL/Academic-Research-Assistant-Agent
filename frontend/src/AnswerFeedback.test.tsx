import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { AnswerFeedbackPanel } from './AnswerFeedback';
import type { AnswerFeedback, FeedbackTarget } from './api';

const target: FeedbackTarget = { schema_version: 1, run_id: 'r1', paper_id: 'p1', question: 'Question?',
  answer: { answerable: true, claims: [{ text: '<script>answer</script>', evidence: [{ chunk_id: 'p1:s0:p0', quote: 'original quote' }] }] },
  prompt_version: 'v1', language: 'zh', evidence: [{ chunk_id: 'p1:s0:p0', version: 'v1', source: 'test', section_index: 0, paragraph_index: 0, text_sha256: 'abc' }] };
const record = (revision = 1, note = 'saved note'): AnswerFeedback => ({ run_id: 'r1', revision, note, rating: 'helpful', target_sha256: 'hash', target, created_at: 'now', updated_at: 'now' });
const json = (feedback: AnswerFeedback | null) => new Response(JSON.stringify({ feedback }));
const fail = (status: number) => new Response(JSON.stringify({ detail: 'failure' }), { status });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('requires an explicit rating, creates and edits the same bound answer with revision checks', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(json(null)).mockResolvedValueOnce(json(record())).mockResolvedValueOnce(json(record(2, 'changed')));
  vi.stubGlobal('fetch', fetch); render(<AnswerFeedbackPanel runId="r1" target={target} />);
  const submit = await screen.findByRole('button', { name: '提交反馈' });
  expect((submit as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByLabelText('有帮助') as HTMLInputElement).checked).toBe(false);
  fireEvent.click(screen.getByLabelText('有帮助'));
  fireEvent.change(screen.getByLabelText('补充说明（可选）'), { target: { value: 'saved note' } });
  fireEvent.click(submit); await screen.findByText(/反馈已保存（版本 1）/);
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ target, rating: 'helpful', note: 'saved note', expected_revision: 0 });
  fireEvent.change(screen.getByLabelText('补充说明（可选）'), { target: { value: 'changed' } });
  fireEvent.click(screen.getByRole('button', { name: '更新反馈' })); await screen.findByText(/反馈已保存（版本 2）/);
  expect(JSON.parse(fetch.mock.calls[2][1].body).expected_revision).toBe(1);
});

it('preserves an uncertain save for retry and prevents overlapping submits', async () => {
  let resolve!: (response: Response) => void;
  const fetch = vi.fn().mockResolvedValueOnce(json(null)).mockImplementationOnce(() => new Promise<Response>(r => { resolve = r; })).mockResolvedValueOnce(json(record()));
  vi.stubGlobal('fetch', fetch); render(<AnswerFeedbackPanel runId="r1" target={target} />);
  await screen.findByRole('button', { name: '提交反馈' }); fireEvent.click(screen.getByLabelText('有帮助'));
  fireEvent.change(screen.getByLabelText('补充说明（可选）'), { target: { value: 'draft' } });
  fireEvent.click(screen.getByRole('button', { name: '提交反馈' }));
  expect((screen.getByRole('button', { name: '正在保存反馈…' }) as HTMLButtonElement).disabled).toBe(true);
  await act(async () => resolve(fail(503)));
  expect(screen.getByRole('alert').textContent).toContain('保存结果尚未确认');
  expect((screen.getByLabelText('补充说明（可选）') as HTMLTextAreaElement).value).toBe('draft');
  fireEvent.click(screen.getByRole('button', { name: '提交反馈' })); await screen.findByText(/反馈已保存/);
  expect(fetch.mock.calls[1][1].body).toBe(fetch.mock.calls[2][1].body);
});

it('blocks stale edits until explicit reload and then uses the latest revision', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(json(record())).mockResolvedValueOnce(fail(409)).mockResolvedValueOnce(json(record(2, 'other editor'))).mockResolvedValueOnce(json(record(3)));
  vi.stubGlobal('fetch', fetch); render(<AnswerFeedbackPanel runId="r1" target={target} />);
  fireEvent.click(await screen.findByRole('button', { name: '更新反馈' })); await screen.findByRole('alert');
  expect((screen.getByRole('button', { name: '更新反馈' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: /重新读取已保存反馈/ }));
  await screen.findByDisplayValue('other editor');
  fireEvent.click(screen.getByLabelText('存在问题')); fireEvent.click(screen.getByRole('button', { name: '更新反馈' }));
  await screen.findByText(/反馈已保存（版本 3）/);
  expect(JSON.parse(fetch.mock.calls[3][1].body).expected_revision).toBe(2);
});

it.each([401, 403, 404])('clears private saved content and editing on access failure %s', async status => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(json(record())).mockResolvedValueOnce(fail(status)));
  render(<AnswerFeedbackPanel runId="r1" />); fireEvent.click(await screen.findByRole('button', { name: '更新反馈' }));
  await screen.findByRole('alert');
  expect(screen.queryByText('<script>answer</script>')).toBeNull();
  expect(screen.queryByRole('textbox')).toBeNull();
});

it('loads the historical published snapshot as escaped text and exposes version provenance', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(record())));
  const { container } = render(<AnswerFeedbackPanel runId="r1" />);
  await screen.findByText('<script>answer</script>');
  expect(container.querySelector('script')).toBeNull();
  expect(screen.getByText('original quote')).toBeTruthy();
  expect(screen.getByText(/论文版本 v1/)).toBeTruthy();
});

it('does not invent a historical answer from metadata when no feedback was submitted', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(null))); render(<AnswerFeedbackPanel runId="r1" />);
  await screen.findByText(/历史元数据无法还原/); expect(screen.queryByRole('textbox')).toBeNull();
});

it('aborts saving on identity unmount and ignores the late response', async () => {
  let resolve!: (response: Response) => void; let signal!: AbortSignal;
  const fetch = vi.fn().mockResolvedValueOnce(json(record())).mockImplementationOnce((_url, options) => {
    signal = options.signal; return new Promise<Response>(r => { resolve = r; });
  }).mockResolvedValueOnce(json(null));
  vi.stubGlobal('fetch', fetch);
  const { rerender } = render(<AnswerFeedbackPanel key="alice" runId="r1" target={target} />);
  fireEvent.click(await screen.findByRole('button', { name: '更新反馈' }));
  rerender(<AnswerFeedbackPanel key="bob" runId="r2" />);
  expect(signal.aborted).toBe(true); await act(async () => resolve(json(record(2, 'private late note'))));
  await screen.findByText(/历史元数据无法还原/);
  expect(screen.queryByText(/private late note/)).toBeNull();
});
