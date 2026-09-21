import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadContext, request } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('request failure boundaries', () => {
  it('preserves server detail without returning failed response as data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '语料正在更新，请稍后重试检索' }), { status: 409 })));
    await expect(request('/api/v1/retrieve')).rejects.toThrow('语料正在更新');
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
