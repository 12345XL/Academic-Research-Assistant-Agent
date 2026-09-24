import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { PdfUpload } from './PdfUpload';
import { setAccessToken } from './api';

const result = { outcome:'imported', paper:{paper_id:'pdf-one',title:'Paper',source:'pdf'}, page_count:2, paragraphs:2,
  warnings:['双栏、表格尚未识别'],preview:[{chunk_id:'p1',section_name:'PDF 第 1 页',text:'Synthetic method.'}] };
function choose(file=new File(['%PDF-synthetic'],'测试.pdf',{type:'application/pdf'})) {
  fireEvent.click(screen.getByText('上传文本 PDF'));
  fireEvent.change(screen.getByLabelText('选择 PDF 文件'),{target:{files:[file]}});
  return file;
}
afterEach(()=>{cleanup(); setAccessToken(''); vi.unstubAllGlobals();});

it('uploads binary PDF with credentials, reports provenance and opens only on explicit click',async()=>{
  const fetch=vi.fn().mockResolvedValue(new Response(JSON.stringify(result)));vi.stubGlobal('fetch',fetch);
  setAccessToken('token'); const open=vi.fn(); render(<PdfUpload onOpen={open}/>);const file=choose();
  fireEvent.click(screen.getByRole('button',{name:'上传并解析'}));await screen.findByText(/已完成入库/);
  expect(fetch.mock.calls[0][0]).toBe('/api/v1/uploads/pdf');
  expect(fetch.mock.calls[0][1].body).toBe(file);
  expect(fetch.mock.calls[0][1].headers.get('X-PDF-Filename')).toBe(encodeURIComponent('测试.pdf'));
  expect(fetch.mock.calls[0][1].headers.get('Authorization')).toBe('Bearer token');
  expect(open).not.toHaveBeenCalled();expect(screen.getByText('双栏、表格尚未识别')).toBeTruthy();
  fireEvent.click(screen.getByText('核对提取文字样例'));expect(screen.getByText('Synthetic method.')).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:'打开论文并提问'}));expect(open).toHaveBeenCalledWith(result.paper);
});

it('blocks empty and oversized files before transmission',()=>{
  const fetch=vi.fn();vi.stubGlobal('fetch',fetch);render(<PdfUpload onOpen={vi.fn()}/>);
  choose(new File([],'empty.pdf'));fireEvent.click(screen.getByRole('button',{name:'上传并解析'}));
  expect(screen.getByRole('alert').textContent).toContain('非空');expect(fetch).not.toHaveBeenCalled();
  const big=new File(['x'],'big.pdf');Object.defineProperty(big,'size',{value:11*1024*1024});
  fireEvent.change(screen.getByLabelText('选择 PDF 文件'),{target:{files:[big]}});
  fireEvent.click(screen.getByRole('button',{name:'上传并解析'}));expect(fetch).not.toHaveBeenCalled();
});

it('prevents duplicate in-flight uploads and makes uncertain outcomes explicit',async()=>{
  let resolve!: (response:Response)=>void;
  const fetch=vi.fn().mockImplementation(()=>new Promise<Response>(r=>{resolve=r;}));vi.stubGlobal('fetch',fetch);
  render(<PdfUpload onOpen={vi.fn()}/>);choose();fireEvent.click(screen.getByRole('button',{name:'上传并解析'}));
  expect((screen.getByRole('button',{name:'正在解析并入库…'}) as HTMLButtonElement).disabled).toBe(true);
  await act(async()=>resolve(new Response(JSON.stringify({detail:'存储失败'}),{status:503})));
  expect(screen.getByRole('alert').textContent).toContain('同一文件');expect(screen.queryByText('打开论文并提问')).toBeNull();
});

it('clears stale previews when selecting another file',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({...result,outcome:'skipped'}))));
  render(<PdfUpload onOpen={vi.fn()}/>);choose();fireEvent.click(screen.getByRole('button',{name:'上传并解析'}));
  await screen.findByText(/已复用已有论文/);
  fireEvent.change(screen.getByLabelText('选择 PDF 文件'),{target:{files:[new File(['next'],'next.pdf')]}});
  expect(screen.queryByText(/已复用已有论文/)).toBeNull();
});

it('aborts on identity unmount and never opens a late uploaded paper',async()=>{
  let resolve!: (response:Response)=>void;let signal!:AbortSignal;
  vi.stubGlobal('fetch',vi.fn().mockImplementation((_url,options)=>{signal=options.signal;return new Promise<Response>(r=>{resolve=r;});}));
  const open=vi.fn();const view=render(<PdfUpload key="old" onOpen={open}/>);choose();fireEvent.click(screen.getByRole('button',{name:'上传并解析'}));
  view.rerender(<PdfUpload key="new" onOpen={open}/>);expect(signal.aborted).toBe(true);
  await act(async()=>resolve(new Response(JSON.stringify(result))));
  expect(screen.queryByText(/已完成入库/)).toBeNull();expect(open).not.toHaveBeenCalled();
});
