import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, downloadSource, loadContext, request, setAccessToken } from './api';

afterEach(() => { setAccessToken(''); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('request failure boundaries', () => {
  it('preserves server detail without returning failed response as data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '语料正在更新，请稍后重试检索' }), { status: 409 })));
    await expect(request('/api/v1/retrieve')).rejects.toMatchObject({ name: 'ApiError', status: 409, message: '语料正在更新，请稍后重试检索', run: undefined });
  });

  it('carries the final run record with an HTTP error and preserves its status', async () => {
    const run = { trace_id: 'conflict-trace', state: 'blocked', reason: 'corpus_changed', terminal_stage: 'publication_check', latency_ms: 250, stages: [] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '语料已变化，请重试', run }), { status: 409 })));
    const error = await request('/api/v1/answer').catch(error => error);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 409, message: '语料已变化，请重试', run });
  });

  it('distinguishes user cancellation from an unavailable backend', async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new DOMException('Aborted', 'AbortError')).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fetch);
    await expect(request('/api/v1/papers')).rejects.toMatchObject({ name: 'AbortError' });
    await expect(request('/api/v1/papers')).rejects.toThrow('无法连接后端服务');
  });
});

describe('original paragraph context', () => {
  it('loads the following paragraph when a citation lies at a page boundary', async () => {
    const rows = Array.from({ length: 102 }, (_, index) => ({ chunk_id: `c${index}`, text: `p${index}` }));
    const fetch = vi.fn().mockImplementation((url: string) => {
      const offset = Number(new URL(url, 'http://local').searchParams.get('offset'));
      return Promise.resolve(new Response(JSON.stringify({ total: rows.length, items: rows.slice(offset, offset + 100) })));
    });
    vi.stubGlobal('fetch', fetch);
    const context = await loadContext('paper', 'c99', new AbortController().signal);
    expect(context.map(item => item.chunk_id)).toEqual(['c98', 'c99', 'c100']);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('reports a missing citation instead of displaying unrelated context', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ total: 1, items: [{ chunk_id: 'other' }] }))));
    await expect(loadContext('paper', 'missing', new AbortController().signal)).rejects.toThrow('未找到这条证据');
  });
});


describe('in-memory access credential', () => {
  it('adds authorization to reads, posts, and source downloads without putting it in URLs', async () => {
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(new Response('{}')));
    vi.stubGlobal('fetch', fetch);
    vi.stubGlobal('URL', Object.assign(class extends URL {}, { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} }));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    setAccessToken('  private-token  ');
    await request('/api/v1/system');
    await request('/api/v1/answer', undefined, { query: 'method' });
    await downloadSource('p1', new AbortController().signal);
    for (const [url, options] of fetch.mock.calls as [string, RequestInit][]) {
      expect(url).not.toContain('private-token');
      expect(new Headers(options.headers).get('Authorization')).toBe('Bearer private-token');
    }
    setAccessToken('');
    await request('/api/v1/system');
    expect(new Headers(fetch.mock.lastCall?.[1].headers).has('Authorization')).toBe(false);
  });
});
