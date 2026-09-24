"""Append one immutable text PDF using the existing facts, jobs and object store."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import PurePosixPath

from psycopg.types.json import Jsonb

from .ingestion import canonical_json, _complete_job
from .pdf_parser import PARSER_VERSION, parse_pdf
from .storage import StorageError


def pdf_id(data):
    return 'pdf-' + hashlib.sha256(PARSER_VERSION.encode() + b'\0' + data).hexdigest()


def build_pdf_document(data, filename, parsed, title=None):
    ident = pdf_id(data)
    raw_hash = hashlib.sha256(data).hexdigest()
    name = PurePosixPath(filename.replace('\\', '/')).name
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)[:180]
    title = title or (name.removesuffix('.pdf').removesuffix('.PDF').strip() or '上传的 PDF')
    paper = {'paper_id': ident, 'title': title, 'abstract': '', 'split': 'uploaded', 'source': 'pdf',
             'version': PARSER_VERSION + ':' + raw_hash, 'pdf_page_count': parsed['page_count'],
             'pdf_parse_warnings': parsed['warnings']}
    chunks = []
    for page in parsed['pages']:
        for index, text in enumerate(page['blocks']):
            chunks.append({**{k: paper[k] for k in ('paper_id','title','split','source','version')},
                'chunk_id': f'{ident}:s{page["page"]-1}:p{index}', 'section_name': f'PDF 第 {page["page"]} 页',
                'section_index': page['page']-1, 'paragraph_index': index, 'text': text,
                'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    content = canonical_json({'format':'research-paper-json-v1', 'paper':paper, 'paragraphs':chunks})
    return paper, chunks, content


def ingest_pdf(repo, store, data, filename, authorize=lambda: None):
    parsed = parse_pdf(data)
    ident = pdf_id(data)
    raw_hash = hashlib.sha256(data).hexdigest()
    job_id = uuid.uuid4()
    with repo.import_lock() as conn:
        authorize()
        existing = repo.get_paper(ident)
        paper, chunks, content = build_pdf_document(data, filename, parsed, existing['title'] if existing else None)
        derived_hash = hashlib.sha256(content).hexdigest()
        raw_key, derived_key = f'original/pdf/{raw_hash}.pdf', f'derived/pdf/{derived_hash}.json'
        version_id = uuid.uuid5(uuid.NAMESPACE_URL, ident + ':' + derived_hash)
        assets = [(raw_key, data, raw_hash, 'application/pdf'), (derived_key, content, derived_hash, 'application/json')]
        conn.execute("INSERT INTO ingestion_jobs(job_id,source,manifest_sha256,status) VALUES(%s,'pdf',%s,'running')",
                     (job_id, derived_hash))
        try:
            store.ensure_bucket()
            with conn.transaction():
                if conn.execute("SELECT 1 FROM stored_objects WHERE bucket<>%s LIMIT 1", (store.bucket,)).fetchone():
                    raise StorageError('Configured object bucket differs from published data')
                for key, body, digest, content_type in assets:
                    conn.execute("INSERT INTO stored_objects(object_key,bucket,sha256,size_bytes,content_type,state,created_by_job) "
                        "VALUES(%s,%s,%s,%s,%s,'staged',%s) ON CONFLICT(object_key) DO UPDATE SET "
                        "state=CASE WHEN stored_objects.state='published' THEN 'published' ELSE 'staged' END,"
                        "created_by_job=CASE WHEN stored_objects.state='published' THEN stored_objects.created_by_job ELSE excluded.created_by_job END",
                        (key, store.bucket, digest, len(body), content_type, job_id))
            written = sum(store.put_verified(key, body, digest, content_type=kind) for key, body, digest, kind in assets)
            authorize()
            with conn.transaction():
                if existing:
                    _, old_papers, old_chunks = repo.load_paper_snapshot(ident)
                    if old_papers != [paper] or old_chunks != chunks:
                        raise StorageError('Published PDF facts failed integrity check')
                    old_object = repo.get_paper_object(ident)
                    if old_object is None or old_object['sha256'] != derived_hash:
                        raise StorageError('Published PDF source failed integrity check')
                else:
                    conn.execute("INSERT INTO papers(paper_id) VALUES(%s)", (ident,))
                    conn.execute("INSERT INTO paper_versions(version_id,paper_id,dataset_version,source,split,title,abstract,"
                        "object_key,content_sha256,state,paper_payload) VALUES(%s,%s,%s,'pdf','uploaded',%s,'',%s,%s,'active',%s)",
                        (version_id, ident, paper['version'], paper['title'], derived_key, derived_hash, Jsonb(paper)))
                    with conn.cursor() as cursor:
                        cursor.executemany("INSERT INTO paragraphs(version_id,chunk_id,ordinal,section_index,paragraph_index,"
                            "section_name,text,text_sha256,paragraph_payload) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            [(version_id,c['chunk_id'],i,c['section_index'],c['paragraph_index'],c['section_name'],
                              c['text'],c['text_sha256'],Jsonb(c)) for i,c in enumerate(chunks)])
                    conn.execute("INSERT INTO pdf_documents(paper_id,original_object_key,original_sha256,parser_version,page_count,warnings) "
                        "VALUES(%s,%s,%s,%s,%s,%s)", (ident,raw_key,raw_hash,PARSER_VERSION,parsed['page_count'],Jsonb(parsed['warnings'])))
                    conn.execute("UPDATE papers SET current_version_id=%s,in_current_corpus=true WHERE paper_id=%s", (version_id,ident))
                conn.execute("UPDATE stored_objects SET state='published' WHERE object_key=ANY(%s)", ([raw_key,derived_key],))
                result = {'outcome':'skipped' if existing else 'imported','job_id':str(job_id), 'paper_id':ident,
                          'papers':1,'paragraphs':len(chunks),'objects_written':written,'objects_verified':2}
                _complete_job(conn,job_id,result)
            return {**result, 'paper':repo.get_paper(ident), 'warnings':parsed['warnings'],
                    'preview':chunks[:3], 'page_count':parsed['page_count']}
        except Exception:
            try:
                with conn.transaction():
                    conn.execute("UPDATE ingestion_jobs SET status='failed',finished_at=now(),error_code='pdf_publication_failure' WHERE job_id=%s", (job_id,))
                    conn.execute("UPDATE stored_objects SET state='orphaned' WHERE created_by_job=%s AND state='staged'", (job_id,))
            except Exception:
                pass
            raise StorageError(f'PDF publication failed; existing papers are unchanged. Job: {job_id}') from None
