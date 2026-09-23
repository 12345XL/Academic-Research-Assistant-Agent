"""Control-flow regressions with deterministic faults; no model API calls."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from research_agent.api import create_app
from research_agent.generation import AnswerService, GenerationSettings, ModelFailure
from research_agent.harness import AnswerRun, STAGES
from research_agent.service import CorpusChangedError, PersistentEvidenceService
from test_api import write_corpus
from test_generation import CONFIG, DRAFT, FakeModel, run
from test_persistent_service import RepositoryDouble


def stages(result):
    return {s['name']: s for s in result['run']['stages']}


def test_published_path_reuses_trace_and_records_all_gates():
    result = run()
    record = result['run']
    assert record['trace_id'] == result['trace_id']
    assert (record['state'], record['reason'], record['terminal_stage']) == ('completed', 'published', 'publish')
    assert list(stages(result)) == list(STAGES)
    assert all(s['status'] == 'completed' and s['latency_ms'] >= 0 for s in record['stages'])
    assert record['latency_ms'] == result['generation']['latency_ms']


@pytest.mark.parametrize('kind,reason,calls,terminal', [
    ('no_hits', 'no_retrieved_evidence', 0, 'context'),
    ('oversize', 'context_budget_excluded_all', 0, 'context'),
    ('abstain', 'generator_abstained', 1, 'publication_check'),
])
def test_abstention_reasons_distinguish_context_and_generator(kind, reason, calls, terminal):
    repo = RepositoryDouble()
    if kind == 'oversize':
        repo.change_text('alpha ' * 5000)
    model = FakeModel(draft={'answerable': False, 'claims': []})
    result = run(model, repo, query='missingterm' if kind == 'no_hits' else 'alpha')
    assert result['run']['state'] == 'abstained'
    assert result['run']['reason'] == reason
    assert result['run']['terminal_stage'] == terminal
    assert len(model.payloads) == calls
    assert result['claims'] == []
    assert stages(result)['verify']['status'] == 'not_run'
    assert stages(result)['publish']['status'] == 'not_run'
    if kind == 'abstain':
        assert result['generation']['publication_check'] == 'database_revision_and_hash'
        assert stages(result)['publication_check']['status'] == 'completed'


def test_disabled_generation_has_failure_reason_not_evidence_judgment():
    model = FakeModel()
    result = run(model, config=GenerationSettings())
    assert result['run']['reason'] == 'generation_not_configured'
    assert result['run']['state'] == 'failed'
    assert not model.payloads


@pytest.mark.parametrize('kind', ['citation', 'semantic'])
def test_failed_gates_stop_downstream_steps(kind):
    draft = copy.deepcopy(DRAFT)
    if kind == 'citation':
        draft['claims'][0]['evidence'][0]['quote'] = 'not in the paper'
    model = FakeModel(draft=draft, verify={'addresses_question': False, 'verdicts': [{'claim_index': 0, 'supported': True}]})
    result = run(model)
    assert result['run']['state'] == 'blocked'
    assert result['run']['reason'] == ('citation_invalid' if kind == 'citation' else 'semantic_verification_failed')
    assert result['run']['terminal_stage'] == ('citation_check' if kind == 'citation' else 'verify')
    assert len(model.payloads) == (1 if kind == 'citation' else 2)
    assert stages(result)['publication_check']['status'] == 'not_run'
    assert stages(result)['publish']['status'] == 'not_run'
    assert result['claims'] == []


@pytest.mark.parametrize('phase', ['draft', 'verify'])
@pytest.mark.parametrize('code', ['provider_timeout', 'invalid_output', 'provider_connection_error'])
def test_model_fault_has_stage_reason_and_elapsed_time(phase, code):
    result = run(FakeModel(**{phase: ModelFailure(code)}))
    assert result['run']['state'] == 'failed'
    assert result['run']['reason'] == code
    assert result['run']['terminal_stage'] == ('generate' if phase == 'draft' else 'verify')
    assert stages(result)['publish']['status'] == 'not_run'
    assert result['generation']['calls'][-1]['latency_ms'] >= 0
    assert result['claims'] == []


def test_version_race_keeps_completed_calls_and_releases_gate():
    repo = RepositoryDouble()
    def change(call):
        if call == 2:
            repo.current_revision += 1
    service = AnswerService(CONFIG, FakeModel(callback=change))
    with pytest.raises(CorpusChangedError) as caught:
        service.answer(PersistentEvidenceService(repo), 'alpha', 'p1')
    exc = caught.value
    assert exc.answer_run['reason'] == 'corpus_changed'
    assert exc.answer_run['terminal_stage'] == 'publication_check'
    assert exc.answer_run['stages'][-1]['status'] == 'not_run'
    assert exc.answer_generation['model_calls'] == 2
    assert exc.answer_generation['checks']['semantic_support'] == 'passed'
    assert not service._lock.locked()


@pytest.mark.parametrize('fault,status,reason', [
    ('race', 409, 'corpus_changed'), ('hash', 502, 'evidence_integrity_failed'),
    ('unexpected', 500, 'internal_error'),
])
def test_api_fault_keeps_run_without_draft_or_exception_secrets(tmp_path, fault, status, reason):
    write_corpus(tmp_path)
    repo = RepositoryDouble()
    def change(call):
        if call == 2:
            if fault == 'race': repo.current_revision += 1
            elif fault == 'hash': repo.paragraph['text'] = 'tampered'
            else: raise RuntimeError('secret-upstream-body')
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=FakeModel(callback=change)),
                    raise_server_exceptions=False) as client:
        client.app.state.evidence = PersistentEvidenceService(repo)
        response = client.post('/api/v1/answer', json={'paper_id': 'p1', 'query': 'alpha'})
    assert response.status_code == status
    assert response.json()['run']['reason'] == reason
    assert response.json()['generation']['checks']['citation_integrity'] == 'passed'
    assert 'claims' not in response.json()
    assert all(s not in response.text for s in ('secret-upstream-body', 'test-secret', 'alpha evidence', 'GOLD_SECRET'))


def test_retrieval_failure_has_run_before_generation_starts(tmp_path):
    write_corpus(tmp_path)
    model = FakeModel()
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=model)) as client:
        response = client.post('/api/v1/answer', json={'paper_id': 'missing', 'query': 'alpha'})
    assert response.status_code == 404
    record = response.json()['run']
    assert record['trace_id']
    assert record['reason'] == 'paper_not_found' and record['terminal_stage'] == 'retrieve'
    assert all(s['status'] == 'not_run' and s['latency_ms'] is None for s in record['stages'][1:])
    assert not model.payloads


def test_stage_machine_rejects_skipped_gates_and_terminal_reentry():
    record = AnswerRun()
    with pytest.raises(RuntimeError): record.enter('publish')
    for stage in ('retrieve', 'context', 'generate', 'publication_check'):
        record.enter(stage)
    with pytest.raises(RuntimeError): record.enter('publish')
    with pytest.raises(RuntimeError): record.finish('published')
    record.finish('generator_abstained')
    with pytest.raises(RuntimeError): record.enter('publish')
    with pytest.raises(RuntimeError): record.finish('generator_abstained')


def test_stage_timings_measure_executed_work_and_do_not_invent_skipped_times():
    ticks = iter([0, 0, 0.1, 0.3])
    record = AnswerRun(clock=lambda: next(ticks))
    record.enter('retrieve')
    record.enter('context')
    result = record.finish('no_retrieved_evidence')
    assert result['latency_ms'] == 300
    assert [s['latency_ms'] for s in result['stages']] == [100, 200, None, None, None, None, None]
    assert 'running' not in json.dumps(result)
