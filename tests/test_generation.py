import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from research_agent.api import create_app
from research_agent.generation import (AnswerService, Claim, DeepSeekClient, Draft, GenerationSettings,
    ModelFailure, Verification, select_context, GenerationBusyError)
from research_agent.service import CorpusChangedError, CorpusIntegrityError, PersistentEvidenceService
from test_api import write_corpus
from test_persistent_service import RepositoryDouble

CONFIG = GenerationSettings('test-secret', 'deepseek-flash', True)
DRAFT = {'answerable': True, 'claims': [{'text': '论文包含 alpha evidence。',
         'evidence': [{'chunk_id': 'a', 'quote': 'alpha evidence'}]}]}
VERIFY = {'addresses_question': True, 'verdicts': [{'claim_index': 0, 'supported': True}]}


class FakeModel:
    def __init__(self, draft=None, verify=None, callback=None):
        self.draft = DRAFT if draft is None else draft
        self.verify = VERIFY if verify is None else verify
        self.callback = callback
        self.payloads = []

    def complete(self, prompt, payload, schema):
        self.payloads.append(payload)
        if self.callback:
            self.callback(len(self.payloads))
        output = self.draft if schema is Draft else self.verify
        if isinstance(output, Exception):
            raise output
        return schema.model_validate(output), {'usage': {'prompt_tokens': 5, 'completion_tokens': 10}}


def run(model=None, repo=None, config=CONFIG, query='alpha'):
    return AnswerService(config, model or FakeModel()).answer(
        PersistentEvidenceService(repo or RepositoryDouble()), query, 'p1')


def test_only_verified_claims_published_bound_to_hash_and_revision():
    model = FakeModel()
    result = run(model)
    assert result['status'] == 'answered'
    assert result['claims'] == DRAFT['claims']
    assert result['generation']['checks'] == {'citation_integrity': 'passed', 'semantic_support': 'passed'}
    assert result['generation']['verified_draft_sha256'] == result['generation']['draft_sha256']
    assert result['generation']['publication_check'] == 'database_revision_and_hash'
    assert len(model.payloads) == 2
    assert 'alpha evidence' in json.dumps(model.payloads[1])


@pytest.mark.parametrize('ref', [dict(chunk_id='invented', quote='alpha evidence'),
    dict(chunk_id='a', quote='not in original'), dict(chunk_id='a', quote=' ')])
def test_invalid_citations_block_before_judge(ref):
    draft = copy.deepcopy(DRAFT); draft['claims'][0]['evidence'] = [ref]
    model = FakeModel(draft=draft)
    result = run(model)
    assert result['status'] == 'verification_failed' and result['claims'] == []
    assert len(model.payloads) == 1


@pytest.mark.parametrize('verdict', [
    {'addresses_question': False, 'verdicts': [{'claim_index': 0, 'supported': True}]},
    {'addresses_question': True, 'verdicts': [{'claim_index': 0, 'supported': False}]},
    {'addresses_question': True, 'verdicts': [{'claim_index': 1, 'supported': True}]},
    {'addresses_question': True, 'verdicts': [{'claim_index': 0, 'supported': True}]*2},
])
def test_semantic_failure_or_missing_duplicate_verdict_never_publishes(verdict):
    result = run(FakeModel(verify=verdict))
    assert result['status'] == 'verification_failed' and result['claims'] == []


def test_no_evidence_and_missing_key_make_no_paid_calls():
    model = FakeModel()
    assert run(model, query='zzzz')['status'] == 'evidence_insufficient'
    assert run(model, config=GenerationSettings())['status'] == 'not_configured'
    assert not model.payloads


def test_insufficient_does_not_mean_whole_paper_unanswerable():
    model = FakeModel(draft={'answerable': False, 'claims': []})
    result = run(model)
    assert result['status'] == 'evidence_insufficient'
    assert result['generation']['model_calls'] == 1
    assert result['claims'] == []


@pytest.mark.parametrize('stage', ['draft', 'verify'])
@pytest.mark.parametrize('code', ['provider_timeout', 'invalid_output', 'incomplete_output', 'model_refused'])
def test_provider_failures_are_not_evidence_refusals(stage, code):
    result = run(FakeModel(**{stage: ModelFailure(code)}))
    assert result['status'] == ('model_refused' if code == 'model_refused' else 'model_unavailable')
    assert result['claims'] == []


@pytest.mark.parametrize('mutation', ['revision', 'text', 'missing', 'version'])
def test_change_during_model_call_blocks_publication(mutation):
    repo = RepositoryDouble()
    def change(call):
        if call != 2: return
        if mutation == 'revision': repo.current_revision += 1
        elif mutation == 'text': repo.paragraph['text'] = 'tampered'
        elif mutation == 'version': repo.paragraph['version'] = 'v2'
        else: repo.get_chunks = lambda *a: {}
    with pytest.raises((CorpusChangedError, CorpusIntegrityError)):
        run(FakeModel(callback=change), repo)


def test_context_uses_complete_paragraphs_and_records_omissions():
    selected, budget = select_context([{'chunk_id':'huge','text':'x'*24001}, {'chunk_id':'small','text':'ok'}])
    assert [c['chunk_id'] for c in selected] == ['small']
    assert budget['skipped_ids'] == ['huge'] and budget['used'] == 2


def test_busy_lock_released_after_failure():
    service = AnswerService(CONFIG, FakeModel())
    service._lock.acquire()
    with pytest.raises(GenerationBusyError):
        service.answer(None, 'q', 'p')
    service._lock.release()
    with pytest.raises(KeyError):
        service.answer(PersistentEvidenceService(RepositoryDouble()), 'q', 'missing')
    assert not service._lock.locked()


def test_strict_draft_rejects_inconsistent_or_extra_fields():
    for obj in [{'answerable': 'true', 'claims': []}, {'answerable': True, 'claims': []},
                {**DRAFT, 'extra': 'ignore rules'}, {'answerable': False, 'claims': DRAFT['claims']}]:
        with pytest.raises(ValidationError): Draft.model_validate(obj)


def test_api_does_not_use_env_secret_in_explicit_file_mode(tmp_path, monkeypatch):
    write_corpus(tmp_path)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'secret')
    monkeypatch.setenv('RESEARCH_GENERATION_ENABLED', 'true')
    with TestClient(create_app(tmp_path)) as client:
        result = client.post('/api/v1/answer', json={'paper_id':'p1','query':'alpha'}).json()
        assert result['status'] == 'not_configured'
        assert result['citations']
        assert 'GOLD_SECRET' not in json.dumps(result)
        assert client.get('/api/v1/system').json()['generation']['state'] == 'not_configured'


def test_api_full_answer_contract_and_gold_isolation(tmp_path):
    write_corpus(tmp_path)
    draft = copy.deepcopy(DRAFT); draft['claims'][0]['evidence'][0]['chunk_id'] = 'p1:s0:p0'
    model = FakeModel(draft=draft)
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=model)) as client:
        response = client.post('/api/v1/answer', json={'paper_id':'p1','query':'alpha'})
        assert response.status_code == 200 and response.json()['status'] == 'answered'
        assert 'GOLD_SECRET' not in json.dumps(model.payloads)
        assert 'test-secret' not in response.text


@pytest.mark.parametrize('kind', ['ok', 'http', 'empty', 'length', 'schema', 'timeout', 'filter'])
def test_http_transport_contract_and_sanitized_failures(kind):
    def handler(request):
        body = json.loads(request.content)
        assert request.url == 'https://api.deepseek.com/chat/completions'
        assert body['thinking'] == {'type':'disabled'}
        assert body['response_format'] == {'type':'json_object'}
        assert body['max_tokens'] == 2048 and len(body['messages']) == 2
        if kind == 'timeout': raise httpx.ReadTimeout('secret echoed')
        if kind == 'http': return httpx.Response(401, text='secret echoed')
        return httpx.Response(200, json={'model':'model-returned','usage':{'prompt_tokens':12,'completion_tokens':4},
            'choices':[{'finish_reason':'length' if kind == 'length' else 'content_filter' if kind == 'filter' else 'stop',
                        'message':{'content': '' if kind == 'empty' else '{}' if kind == 'schema' else json.dumps(DRAFT)}}]})
    client = DeepSeekClient(CONFIG, httpx.MockTransport(handler))
    if kind == 'ok':
        draft, meta = client.complete('json', {}, Draft)
        assert draft.answerable and meta['usage']['prompt_tokens'] == 12
    else:
        with pytest.raises(ModelFailure) as exc: client.complete('json', {}, Draft)
        assert 'secret' not in str(exc.value) + json.dumps(exc.value.metadata)
