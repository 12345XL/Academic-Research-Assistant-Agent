"""Real isolated PostgreSQL/S3 ingestion and existing answer/feedback integration."""
import copy
import hashlib
import json
import os

from fastapi.testclient import TestClient
import pytest

from research_agent.access import AccessPolicy
from research_agent.api import create_app
from research_agent.ingestion import ingest_qasper
from research_agent.pdf_ingestion import ingest_pdf, pdf_id
from research_agent.storage import StorageError
from test_generation import CONFIG, DRAFT, FakeModel
from test_pdf import make_pdf, METHOD
from test_storage_integration import infrastructure, corpus

pytestmark = pytest.mark.skipif(os.getenv('RUN_STORAGE_INTEGRATION') != '1', reason='requires local PostgreSQL/S3')


def test_upload_existing_qa_feedback_download_restart_and_qasper_coexist(infrastructure, tmp_path):
    settings, repo, store = infrastructure
    corpus(tmp_path); ingest_qasper(repo,store,tmp_path)
    baseline = repo.load_snapshot(); data = make_pdf(); ident = pdf_id(data)
    draft = copy.deepcopy(DRAFT)
    draft['claims'][0]['text'] = 'Synthetic fixture: uses context-free grammar.'
    draft['claims'][0]['evidence'] = [{'chunk_id':ident+':s0:p0','quote':'Our method uses a context-free grammar'}]
    model = FakeModel(draft=draft)
    with TestClient(create_app(settings=settings,generation_settings=CONFIG,model_client=model)) as client:
        response = client.post('/api/v1/uploads/pdf',content=data,headers={'Content-Type':'application/pdf','X-PDF-Filename':'sample.pdf'})
        assert response.status_code == 200, response.text
        saved=response.json(); assert saved['paper']['paper_id']==ident and saved['page_count']==2
        assert saved['preview'][0]['section_index']==0
        assert repo.load_snapshot()==baseline  # QASPER cache/index membership unchanged
        assert repo.summary()['papers']==2 and repo.summary()['paragraphs']==3
        assert client.get('/api/v1/papers').json()['total']==2
        assert client.get('/api/v1/system').json()['capabilities']['pdf_upload']
        duplicate=client.post('/api/v1/uploads/pdf',content=data,headers={'Content-Type':'application/pdf','X-PDF-Filename':'renamed.pdf'}).json()
        assert duplicate['outcome']=='skipped' and duplicate['paper']['title']=='sample'
        assert duplicate['objects_written']==0
        for mode in ['dense','hybrid']:
            for endpoint in ['retrieve','answer']:
                assert client.post('/api/v1/'+endpoint,json={'paper_id':ident,'query':'method','mode':mode}).status_code==409
        retrieved=client.post('/api/v1/retrieve',json={'paper_id':ident,'query':'method'}).json()
        assert retrieved['citations'][0]['text']==METHOD and retrieved['citations'][0]['source']=='pdf'
        answer=client.post('/api/v1/answer',json={'paper_id':ident,'query':'method'}).json()
        assert answer['status']=='answered',answer
        assert answer['claims']==draft['claims']
        url='/api/v1/runs/'+answer['trace_id']+'/feedback'
        feedback={'target':answer['feedback_target'],'rating':'helpful','expected_revision':0}
        assert client.post(url,json=feedback).status_code==200
        raw=client.get('/api/v1/papers/'+ident+'/pdf')
        assert raw.content==data and raw.headers['cache-control']=='no-store'
        assert raw.headers['content-disposition'].startswith('attachment;')
        assert client.get('/api/v1/papers/'+ident+'/source').json()['paragraphs'][0]['text']==METHOD
        assert len(model.payloads)==2
    # Re-publishing QASPER may change its revision, never hide uploaded documents.
    corpus(tmp_path,'changed alpha evidence'); ingest_qasper(repo,store,tmp_path)
    with TestClient(create_app(settings=settings)) as client:
        assert client.get('/api/v1/papers/'+ident).status_code==200
        assert client.get(url).json()['feedback']['target']['evidence'][0]['section_index']==0
        assert client.post('/api/v1/retrieve',json={'paper_id':ident,'query':'method'}).json()['citations'][0]['text']==METHOD
    with repo.connect() as conn:
        assert conn.execute("SELECT content_type FROM stored_objects WHERE object_key LIKE 'original/pdf/%%'").fetchone()['content_type']=='application/pdf'


@pytest.mark.parametrize('failure', ['object', 'publication'])
def test_failed_upload_has_no_partial_paper_and_retry_is_safe(infrastructure,tmp_path,monkeypatch,failure):
    settings,repo,store=infrastructure; corpus(tmp_path); ingest_qasper(repo,store,tmp_path)
    data=make_pdf(); ident=pdf_id(data); baseline=repo.load_snapshot()
    with monkeypatch.context() as patch:
        if failure=='object':
            put=store.put_verified
            def failing_put(key,body,digest,content_type='application/json'):
                if key.startswith('derived/pdf/'): raise StorageError('injected')
                return put(key,body,digest,content_type)
            patch.setattr(store,'put_verified',failing_put)
        else:
            def fail(*args): raise RuntimeError('injected after insert')
            patch.setattr('research_agent.pdf_ingestion._complete_job',fail)
        with pytest.raises(StorageError): ingest_pdf(repo,store,data,'sample.pdf')
    assert repo.get_paper(ident) is None and repo.load_snapshot()==baseline
    assert repo.list_jobs()[0]['status']=='failed'
    with repo.connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM stored_objects WHERE state='staged'").fetchone()['n']==0
    result=ingest_pdf(repo,store,data,'sample.pdf'); assert result['outcome']=='imported'
    assert repo.summary()['papers']==2


def test_pdf_access_invalid_upload_and_checksum_repair(infrastructure,tmp_path):
    settings,repo,store=infrastructure; data=make_pdf(); ident=pdf_id(data)
    with TestClient(create_app(settings=settings)) as client:
        assert client.post('/api/v1/uploads/pdf',content=data,headers={'Content-Type':'text/plain'}).status_code==415
        assert client.post('/api/v1/uploads/pdf',content=b'wrong',headers={'Content-Type':'application/pdf'}).status_code==422
        assert repo.list_jobs()==[]
    ingest_pdf(repo,store,data,'sample.pdf')
    digest=hashlib.sha256(b'token').hexdigest(); path=tmp_path/'policy.json'
    path.write_text(json.dumps({'tokens':{digest:{'id':'alice','allowed_paper_ids':['p1']}}}))
    with TestClient(create_app(settings=settings,access_policy=AccessPolicy(path))) as client:
        client.headers['Authorization']='Bearer token'
        for endpoint in ['', '/pdf','/source','/paragraphs']:
            assert client.get('/api/v1/papers/'+ident+endpoint).status_code==403
        assert client.get('/api/v1/papers').json()['total']==0
        assert client.post('/api/v1/uploads/pdf',content=data,headers={'Content-Type':'application/pdf'}).status_code==403
        path.write_text(json.dumps({'tokens':{digest:{'id':'alice','allowed_paper_ids':[ident]}}}))
        assert client.get('/api/v1/papers/'+ident+'/pdf').content==data
        assert not client.get('/api/v1/system').json()['capabilities']['pdf_upload']
    key=repo.get_pdf_object(ident)['object_key']
    store.client.put_object(Bucket=store.bucket,Key=key,Body=b'corrupted')
    with TestClient(create_app(settings=settings)) as client:
        assert client.get('/api/v1/papers/'+ident+'/pdf').status_code==502
    assert ingest_pdf(repo,store,data,'sample.pdf')['objects_written']==1
    assert store.read_bytes(key)==data


def test_upload_gate_rejects_overlap_until_actual_worker_finishes(infrastructure,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    settings,repo,store=infrastructure
    entered,release=threading.Event(),threading.Event()
    real=ingest_pdf
    def delayed(*args,**kwargs):
        entered.set(); assert release.wait(10)
        return real(*args,**kwargs)
    monkeypatch.setattr('research_agent.api.ingest_pdf',delayed)
    with TestClient(create_app(settings=settings)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        first=pool.submit(client.post,'/api/v1/uploads/pdf',content=make_pdf(),headers={'Content-Type':'application/pdf'})
        try:
            assert entered.wait(3)
            assert client.post('/api/v1/uploads/pdf',content=make_pdf(),headers={'Content-Type':'application/pdf'}).status_code==409
        finally: release.set()
        assert first.result(timeout=10).status_code==200
        assert repo.summary()['papers']==1
