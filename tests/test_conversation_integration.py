"""Real PostgreSQL, isolated schema and fake model; no production corpus writes."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from research_agent.api import create_app
from research_agent.conversation import ConversationError, PostgresConversationStore
from test_api import write_corpus
from test_conversation import create, ask
from test_feedback import configured_model
from test_generation import CONFIG
from test_run_store_integration import run_repository

pytestmark = pytest.mark.skipif(os.getenv('RUN_STORAGE_INTEGRATION') != '1', reason='requires local PostgreSQL')


@pytest.fixture
def store(run_repository):
    migration = Path(__file__).resolve().parents[1] / 'src/research_agent/migrations/008_conversations.sql'
    with run_repository.connect() as conn:
        conn.execute(migration.read_text())
    return PostgresConversationStore(run_repository)


def test_postgres_two_turns_restart_cas_delete_and_expiry(store, tmp_path):
    write_corpus(tmp_path)
    with TestClient(create_app(tmp_path, generation_settings=CONFIG, model_client=configured_model(),
                               conversation_store=store)) as client:
        created = create(client)
        assert client.get('/api/v1/conversations/' + created['conversation_id']).json() == created
        first = ask(client, created).json()
        second = ask(client, first['conversation'], 'What are its limitations?').json()
        assert second['status'] == 'answered'
        record = second['conversation']; ident = record['conversation_id']
    restarted = PostgresConversationStore(store.repository)
    with TestClient(create_app(tmp_path, conversation_store=restarted)) as client:
        assert client.get('/api/v1/conversations/' + ident).json() == record
    original = restarted.get(ident, 'local-user')
    assert restarted.get(ident, 'other-user') is None
    def append(_):
        try: return restarted.append(original, 'alpha', second)
        except ConversationError: return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(r is not None for r in pool.map(append, range(2))) == 1
    restarted.delete(ident, 'other-user')
    assert restarted.get(ident, 'local-user')
    restarted.delete(ident, 'local-user')
    with pytest.raises(ConversationError): restarted.append(original, 'alpha', second)
    assert restarted.get(ident, 'local-user') is None
    expired = restarted.create('alice', 'p1', 'v1')
    with restarted.repository.connect() as conn:
        conn.execute("UPDATE conversations SET expires_at=clock_timestamp()-interval '1 second'")
    assert restarted.get(expired['conversation_id'], 'alice') is None
    with restarted.repository.connect() as conn:
        assert conn.execute('SELECT count(*) AS n FROM conversations').fetchone()['n'] == 0
