"""Small synthetic PDF fixtures; no downloaded papers or paid model calls."""
import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from research_agent.pdf_parser import parse_pdf, PdfError, MAX_BYTES
from research_agent.pdf_ingestion import build_pdf_document, pdf_id

METHOD = 'Our method uses a context-free grammar to parse text into semantic trees. The grammar and trees jointly identify classes and instances. Human reviewers inspect induced rules.'
EXPERIMENT = 'This synthetic experiment contains 100 fictional sentences and does not report any real research result. It tests page provenance and immutable source storage.'


def make_pdf(pages=(METHOD, EXPERIMENT), encrypted=False):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        content = DecodedStreamObject()
        safe = text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
        content.set_data(('BT /F1 12 Tf 50 750 Td ('+safe+') Tj ET').encode())
        page[NameObject('/Contents')] = writer._add_object(content)
    if encrypted: writer.encrypt('test-password')
    stream = io.BytesIO(); writer.write(stream)
    return stream.getvalue()


def test_parser_page_provenance_stable_hash_and_no_cross_page_chunk():
    data = make_pdf(); parsed = parse_pdf(data)
    assert parsed['page_count'] == 2 and parsed['pages'][0]['blocks'] == [METHOD]
    paper, chunks, content = build_pdf_document(data, '../../demo.pdf', parsed)
    assert paper['title'] == 'demo' and paper['split'] == 'uploaded'
    assert [p['section_index'] for p in chunks] == [0,1]
    assert chunks[1]['section_name'] == 'PDF 第 2 页'
    assert pdf_id(data) != pdf_id(make_pdf((METHOD+' changed',)))
    assert build_pdf_document(data,'demo.pdf',parsed)[2] == content


@pytest.mark.parametrize('data,code', [(b'', 'size_limit'), (b'not pdf', 'invalid_pdf'),
    (b'%PDF-1.4\nbroken', 'parse_failed'), (make_pdf(('',)), 'no_text'),
    (make_pdf(encrypted=True), 'encrypted_pdf'), (make_pdf(('',)*81), 'page_limit'),
    (b'%PDF-'+b'0'*MAX_BYTES, 'size_limit'), (make_pdf(('x'*500001,)), 'text_limit')])
def test_rejects_unsupported_and_bounded_inputs(data, code):
    with pytest.raises(PdfError) as failure: parse_pdf(data)
    assert failure.value.code == code


def test_mixed_empty_pages_are_reported_and_never_invented():
    result = parse_pdf(make_pdf((METHOD,'')))
    assert result['pages'][1]['blocks'] == []
    assert '2' in result['warnings'][1] and 'OCR' in result['warnings'][1]


def test_parser_timeout_terminates_worker():
    with pytest.raises(PdfError) as failure: parse_pdf(make_pdf(), timeout=.0001)
    assert failure.value.code == 'parse_timeout'


def test_long_text_is_bounded_into_chunks_without_losing_words():
    original = 'context grammar method ' * 200
    result = parse_pdf(make_pdf((original,)))
    blocks = result['pages'][0]['blocks']
    assert all(len(block) <= 1600 for block in blocks)
    assert ' '.join(blocks) == original.strip()


def test_parser_works_outside_repository_without_pythonpath(monkeypatch, tmp_path):
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.chdir(tmp_path)
    assert parse_pdf(make_pdf())['page_count'] == 2
