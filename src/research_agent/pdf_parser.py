"""Bounded text-only PDF extraction in a disposable process; no OCR or execution."""
from __future__ import annotations

import io
from pathlib import Path
import json
import logging
import re
import subprocess
import sys
import time

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 80
MAX_CHARS = 500_000
PARSER_VERSION = 'pypdf-6.19.0-plain-chunks-v1'


class PdfError(ValueError):
    def __init__(self, code, message, status=422):
        self.code, self.status = code, status
        super().__init__(message)


def extract_text(data: bytes) -> dict:
    from pypdf import PdfReader
    if not data.startswith(b'%PDF-'):
        raise PdfError('invalid_pdf', '文件不是有效的 PDF')
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise PdfError('encrypted_pdf', '暂不支持加密 PDF，请提供未加密的文本 PDF')
    if not 1 <= len(reader.pages) <= MAX_PAGES:
        raise PdfError('page_limit', 'PDF 须为 1–80 页')
    pages, sparse, total = [], [], 0
    for number, page in enumerate(reader.pages, 1):
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > 5_000_000:
            raise PdfError('page_too_complex', 'PDF 页面内容过大，请拆分后再上传')
        text = page.extract_text() or ''
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t\r')
        if '\ufffd' in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
            raise PdfError('invalid_text', 'PDF 存在无法解码的文字，请核对字体或换用文本版本')
        total += len(text)
        if total > MAX_CHARS:
            raise PdfError('text_limit', '提取文字超过 50 万字符，请拆分文件')
        if len(re.sub(r'\s', '', text)) < 40:
            sparse.append(number)
        # Preserve page boundaries; these are extraction chunks, not inferred sections.
        blocks = []
        for paragraph in re.split(r'\n\s*\n', text.strip()):
            paragraph = re.sub(r'\s+', ' ', paragraph).strip()
            while paragraph:
                end = min(len(paragraph), 1600)
                if end < len(paragraph):
                    boundary = paragraph.rfind(' ', 800, end)
                    if boundary >= 800: end = boundary
                blocks.append(paragraph[:end]); paragraph = paragraph[end:].strip()
        pages.append({'page': number, 'blocks': blocks})
    if sum(len(re.sub(r'\s', '', b)) for p in pages for b in p['blocks']) < 80:
        raise PdfError('no_text', '未提取到足够文字；扫描件需要 OCR，本阶段暂不支持')
    warnings = ['按 PDF 内容流提取文字；双栏阅读顺序、表格和公式未作结构识别，请对照原 PDF 核对。']
    if sparse:
        warnings.append('以下页文字较少或无文字，可能是扫描页/图页，未做 OCR：' + '、'.join(map(str, sparse)))
    return {'pages': pages, 'page_count': len(pages), 'warnings': warnings, 'parser_version': PARSER_VERSION}


def parse_pdf(data: bytes, timeout=30) -> dict:
    if not data or len(data) > MAX_BYTES:
        raise PdfError('size_limit', 'PDF 文件须非空且不超过 10 MiB', 413)
    if not data.startswith(b'%PDF-'):
        raise PdfError('invalid_pdf', '文件不是有效的 PDF')
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    try:
        first = True
        while True:
            if time.monotonic() >= deadline:
                raise PdfError('parse_timeout', 'PDF 解析超时，已停止；请拆分或换用简单文本 PDF')
            try:
                output, _ = process.communicate(input=data if first else None, timeout=min(.2, max(.01, deadline-time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                first = False
                if sys.platform == 'darwin':
                    # macOS does not support RLIMIT_AS. Sample RSS while the parent
                    # retains a hard wall-clock deadline; this is not an instantaneous cap.
                    usage = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)],
                                           capture_output=True, timeout=1)
                    if usage.stdout.strip() and int(usage.stdout.strip()) * 1024 > 768 * 1024 * 1024:
                        raise PdfError('memory_limit', 'PDF 解析超出内存限制，请拆分文件')
        if process.returncode != 0:
            raise PdfError('parse_failed', 'PDF 损坏、过于复杂或超出解析资源限制')
    finally:
        if process.poll() is None: process.kill()
        process.communicate()
    try:
        parsed = json.loads(output)
    except (ValueError, UnicodeError):
        raise PdfError('parse_failed', 'PDF 解析未能完成') from None
    if 'error' in parsed:
        raise PdfError(parsed['error'], parsed['message'])
    return parsed


def _worker():
    # Local macOS/Linux guard, not a network/container security sandbox.
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    logging.disable(logging.CRITICAL)
    try:
        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES: raise ValueError('limit')
        result = extract_text(data)
    except PdfError as exc:
        result = {'error': exc.code, 'message': str(exc)}
    except Exception:
        result = {'error': 'parse_failed', 'message': 'PDF 损坏、过于复杂或无法解析'}
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    _worker()
