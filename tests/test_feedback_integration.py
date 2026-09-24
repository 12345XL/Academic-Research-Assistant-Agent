"""Feedback survives PostgreSQL restarts; all records live in a temporary schema."""
from concurrent.futures import ThreadPoolExecutor
import copy
import os

from fastapi.testclient import TestClient
import pytest

from research_agent.api import create_app
from research_agent.feedback import FeedbackConflictError, FeedbackInput
from research_agent.run_store import PostgresRunStore
from test_api import write_corpus
from test_feedback import answer, body, configured_model
from test_generation import CONFIG
from test_run_store_integration import run_repository

pytestmark = pytest.mark.skipif(os.getenv('RUN_STORAGE_INTEGRATION') != '1', reason='requires local PostgreSQL')


def test_feedback_restart_provenance_ownership_and_no_text_in_run(run_repository, tmp_path):
    write_corpus(tmp_path)
    store = PostgresRunStore(run_repository)
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=configured_model(), run_store=store)) as client:
        result = answer(client); ident = result['trace_id']; url = f'/api/v1/runs/{ident}/feedback'
        payload = body(result, note='保存后重启仍可追溯')
        response = client.post(url, json=payload)
        assert response.status_code == 200
        saved = response.json()['feedback']
        assert client.post(url, json=payload).json()['feedback'] == saved
        assert store.get_feedback(ident, 'other-user') is None
        with pytest.raises(FeedbackConflictError): store.save_feedback(ident, 'other-user', FeedbackInput(**payload))
        assert 'question' not in store.get(ident, 'local-user')['snapshot']
    restarted = PostgresRunStore(run_repository)
    with TestClient(create_app(tmp_path, run_store=restarted)) as client:
        assert client.get(url).json()['feedback'] == saved
        altered = copy.deepcopy(payload); altered['target']['question'] = 'forged after restart'
        assert client.post(url, json=altered).status_code == 409
        changed = client.post(url, json=body(result, rating='problem', expected_revision=1)).json()['feedback']
        assert changed['revision'] == 2 and changed['target'] == saved['target']
        assert client.post(url, json=payload).status_code == 409
        assert client.get(url).json()['feedback'] == changed


def test_postgres_concurrent_edits_use_revision_guard(run_repository, tmp_path):
    write_corpus(tmp_path); store = PostgresRunStore(run_repository)
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=configured_model(), run_store=store)) as client:
        result = answer(client); ident = result['trace_id']
        store.save_feedback(ident, 'local-user', FeedbackInput(**body(result)))
        def edit(note):
            try:
                return store.save_feedback(ident, 'local-user', FeedbackInput(**body(result, note=note, expected_revision=1)))
            except FeedbackConflictError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(edit, ['tab A', 'tab B']))
        assert sum(item is not None for item in results) == 1
        assert store.get_feedback(ident, 'local-user')['revision'] == 2
