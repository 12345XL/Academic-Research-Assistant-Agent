"""Bounded working memory, distinct from source evidence and run metadata."""
import copy
from datetime import timedelta
import json

from fastapi.testclient import TestClient
import pytest

from research_agent.access import AccessPolicy
from research_agent.api import create_app
from research_agent.conversation import (ConversationError, MemoryConversationStore, MAX_CHARS,
                                         MAX_TURNS, utcnow, validate, working_context)
from research_agent.runtime import RunLimits
from test_access import write_policy, TOKEN
from test_api import write_corpus
from test_feedback import configured_model
from test_generation import CONFIG


def create(client):
    response = client.post('/api/v1/conversations', json={'paper_id': 'p1'})
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    return response.json()


def ask(client, conversation, query='alpha method', **extra):
    return client.post('/api/v1/answer', json=dict(paper_id='p1', query=query,
                       conversation_id=conversation['conversation_id'],
                       conversation_revision=conversation['revision'], **extra))


@pytest.fixture
def session(tmp_path):
    write_corpus(tmp_path)
    store, model = MemoryConversationStore(), configured_model()
    app = create_app(tmp_path, generation_settings=CONFIG, model_client=model, conversation_store=store)
    with TestClient(app) as client:
        yield client, store, model, app


def test_opt_in_two_turns_retrieve_again_both_models_receive_context_and_trace_has_no_text(session):
    client, store, model, app = session
    plain = client.post('/api/v1/answer', json={'paper_id': 'p1', 'query': 'alpha'})
    assert plain.json()['status'] == 'answered' and not store.records
    first = ask(client, create(client)).json()
    second = ask(client, first['conversation'], 'What are its limitations?').json()
    assert second['status'] == 'answered' and second['query'] == 'What are its limitations?'
    assert second['memory_used_turns'] == 1 and len(second['conversation']['turns']) == 2
    assert second['generation']['model_calls'] == 2
    assert model.payloads[-2]['question'] == second['query']
    assert model.payloads[-2]['conversation_context'] == model.payloads[-1]['conversation_context']
    assert model.payloads[-2]['evidence'][0]['text'] == 'alpha evidence'
    assert len(model.payloads[-1]['cited_paragraphs']) == 1
    third = ask(client, second['conversation'], '它还有什么局限？').json()
    assert third['status'] == 'answered'  # original query remains the retrieval anchor
    trace = client.get('/api/v1/runs/' + second['trace_id']).json()
    assert trace['metadata']['conversation_id'] == first['conversation']['conversation_id']
    assert trace['snapshot']['generation']['memory_used_turns'] == 1
    assert 'alpha evidence' not in json.dumps(trace) and 'GOLD_SECRET' not in json.dumps(second)
    fresh = create(client)
    assert not fresh['turns']  # same owner and paper, independently isolated session


def test_missing_context_multiple_claims_and_stale_revision_clarify_without_model_calls(session):
    client, store, model, app = session
    conversation = create(client)
    assert ask(client, conversation, '这个方法有什么局限？').status_code == 409
    assert not model.payloads
    saved = ask(client, conversation).json()['conversation']
    count = len(model.payloads)
    assert ask(client, conversation).status_code == 409
    store.records[saved['conversation_id']]['turns'][-1]['claim_count'] = 2
    assert ask(client, saved, '这个方法有什么局限？').status_code == 409
    assert len(model.payloads) == count
    assert client.get('/api/v1/conversations/' + saved['conversation_id']).json()['revision'] == 1


def test_paper_switch_version_change_and_clear_never_reuse_old_memory(session):
    client, store, model, app = session
    conversation = ask(client, create(client)).json()['conversation']
    payload = {'paper_id': 'p2', 'query': 'alpha', 'conversation_id': conversation['conversation_id'],
               'conversation_revision': 1}
    assert client.post('/api/v1/answer', json=payload).status_code == 409
    app.state.evidence.papers['p1']['version'] = 'changed'
    assert ask(client, conversation).status_code == 409
    assert client.get('/api/v1/conversations/' + conversation['conversation_id']).status_code == 409
    assert client.post('/api/v1/conversations/' + conversation['conversation_id'] + '/clear').status_code == 200
    assert not store.records
    assert ask(client, conversation).status_code == 404
    assert len(model.payloads) == 2


@pytest.mark.parametrize('mutation', ['clear', 'expire', 'version'])
def test_invalidated_during_generation_cannot_publish_or_restore_memory(session, mutation):
    client, store, model, app = session
    conversation = create(client)
    def change(_):
        if mutation == 'clear': store.delete(conversation['conversation_id'], 'local-user')
        elif mutation == 'expire': store.clock = lambda: utcnow() + timedelta(days=2)
        else: app.state.evidence.papers['p1']['version'] = 'changed'
    model.callback = change
    response = ask(client, conversation)
    data = response.json()
    assert data.get('status') != 'answered' and not data.get('claims')
    assert len(model.payloads) == 1
    current = store.get(conversation['conversation_id'], 'local-user')
    assert current is None or not current['turns']


def test_failed_verification_and_memory_only_quote_do_not_enter_memory(session):
    client, store, model, app = session
    conversation = ask(client, create(client)).json()['conversation']
    model.draft = copy.deepcopy(model.draft)
    model.draft['claims'][0]['evidence'][0]['quote'] = '论文包含 alpha evidence。'
    result = ask(client, conversation, 'What are its limitations?').json()
    assert result['status'] == 'verification_failed' and not result['claims']
    assert result['conversation']['revision'] == 1
    assert len(model.payloads) == 3  # invalid quote stopped before semantic verification


def test_memory_storage_failure_keeps_answer_outcome_explicit(session, monkeypatch):
    client, store, model, app = session
    def unavailable(*args): raise OSError('private connection details')
    monkeypatch.setattr(store, 'append', unavailable)
    result = ask(client, create(client)).json()
    assert result['status'] == 'answered' and '保存失败' in result['memory_notice']
    assert 'private connection details' not in json.dumps(result)


def test_owner_isolation_revocation_and_owner_delete_after_paper_revocation(tmp_path):
    write_corpus(tmp_path)
    path = tmp_path / 'policy.json'; write_policy(path)
    model = configured_model()
    with TestClient(create_app(tmp_path, access_policy=AccessPolicy(path), generation_settings=CONFIG,
                               model_client=model)) as client:
        client.headers['Authorization'] = 'Bearer ' + TOKEN
        conversation = ask(client, create(client)).json()['conversation']
        url = '/api/v1/conversations/' + conversation['conversation_id']
        write_policy(path, identity='bob')
        assert client.get(url).status_code == 404
        assert ask(client, conversation).status_code == 404
        client.post(url + '/clear')
        write_policy(path)
        assert client.get(url).status_code == 200
        write_policy(path, papers=[])
        assert client.get(url).status_code == 403 and ask(client, conversation).status_code == 403
        assert client.post(url + '/clear').status_code == 200
        write_policy(path)
        assert client.get(url).status_code == 404


def test_store_bounded_expiry_cas_and_no_resurrection():
    clock = [utcnow()]
    store = MemoryConversationStore(clock=lambda: clock[0])
    item = store.create('alice', 'p1', 'v1')
    stale = copy.deepcopy(item)
    for i in range(12):
        item = store.append(item, 'q' * 1000, {'status': 'answered', 'trace_id': str(i),
                            'claims': [{'text': 'a' * 1000}]})
    assert len(item['turns']) <= MAX_TURNS
    assert sum(len(t['answer']) + len(t['question']) for t in item['turns']) <= MAX_CHARS
    with pytest.raises(ConversationError): store.append(stale, 'q', {'status': 'answered', 'claims': []})
    assert store.get(item['conversation_id'], 'bob') is None
    with pytest.raises(ConversationError): validate(item, 'p2', 'v1')
    clock[0] += timedelta(hours=24)
    assert store.get(item['conversation_id'], 'alice') is None
    with pytest.raises(ConversationError): store.append(item, 'q', {'status': 'answered', 'claims': []})
    assert not store.records


def test_memory_pair_validation_and_client_cannot_supply_history(session):
    client, *_ = session
    for extra in [{'conversation_id': 'a' * 32}, {'conversation_revision': 0}, {'memory_context': []},
                  {'conversation_id': 'a' * 32, 'conversation_revision': True}]:
        assert client.post('/api/v1/answer', json={'paper_id': 'p1', 'query': 'alpha', **extra}).status_code == 422


def test_standalone_question_does_not_reuse_memory_and_all_prompts_still_have_budget(session, tmp_path):
    client, store, model, app = session
    conversation = ask(client, create(client), 'What is the method for alpha?').json()['conversation']
    result = ask(client, conversation, 'alpha dataset').json()
    assert result['memory_used_turns'] == 0
    assert 'conversation_context' not in model.payloads[-2]
    assert 'conversation_context' not in model.payloads[-1]
    assert result['conversation']['expires_at'] == conversation['expires_at']
    model.payloads.clear()
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=model,
                               conversation_store=store, run_limits=RunLimits(max_prompt_chars=100))) as bounded:
        response = ask(bounded, result['conversation'], 'What are its limitations?')
        assert 'input_budget_exceeded' in response.text
        assert not model.payloads
        assert store.get(conversation['conversation_id'], 'local-user')['revision'] == 2


def test_revocation_during_generation_does_not_publish_or_append(tmp_path):
    write_corpus(tmp_path)
    path = tmp_path / 'policy.json'; write_policy(path)
    model, store = configured_model(), MemoryConversationStore()
    model.callback = lambda _: write_policy(path, papers=[])
    with TestClient(create_app(tmp_path, access_policy=AccessPolicy(path), generation_settings=CONFIG,
                               model_client=model, conversation_store=store)) as client:
        client.headers['Authorization'] = 'Bearer ' + TOKEN
        conversation = create(client)
        response = ask(client, conversation)
        assert response.status_code == 403 and not response.json().get('claims')
        assert len(model.payloads) == 1
        assert store.get(conversation['conversation_id'], 'alice')['turns'] == []
