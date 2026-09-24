"""Opt-in, bounded paper working memory. Never a source of factual evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
import threading
import uuid

from psycopg.types.json import Jsonb

MAX_TURNS = 6
MAX_CHARS = 6000
TTL_HOURS = 24
MAX_SESSIONS = 50


class ConversationError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def utcnow():
    return datetime.now(timezone.utc)


def new_conversation(owner, paper, version, now):
    return dict(conversation_id=uuid.uuid4().hex, owner_id=owner, paper_id=paper,
                paper_version=version, revision=0, turns=[],
                expires_at=(now + timedelta(hours=TTL_HOURS)).isoformat())


def validate(record, paper, version, revision=None):
    if record is None:
        raise ConversationError('对话不存在或已过期，请清空后开始新对话。', 404)
    if record['paper_id'] != paper or record['paper_version'] != version:
        raise ConversationError('论文或版本已变化，请清空后开始新对话。')
    if revision is not None and record['revision'] != revision:
        raise ConversationError('对话已更新，请刷新对话后重新提交问题。')
    return record


def append_turn(record, query, result):
    """Only server-produced, published answers enter memory; no raw citations/gold."""
    if result['status'] != 'answered':
        return record
    text = '\n'.join(c['text'] for c in result['claims'])
    turn = dict(run_id=result['trace_id'], question=query, answer=text[:3000],
                answer_truncated=len(text) > 3000, claim_count=len(result['claims']))
    turns = [*record['turns'], turn][-MAX_TURNS:]
    while sum(len(t['question']) + len(t['answer']) for t in turns) > MAX_CHARS:
        turns.pop(0)
    return {**record, 'revision': record['revision'] + 1, 'turns': turns}


_FOLLOWUP = re.compile(r'这个方法|该方法|这种方法|它|上述|前面|刚才|上一个|\b(?:it|its|that method|this method|previous answer)\b|^(?:那|那么|继续|为什么|有什么局限|有哪些局限|局限呢)', re.I)


def working_context(query, record):
    """Conservative routing, not a general semantic reference resolver."""
    if not _FOLLOWUP.search(query):
        return [], query
    turns = record['turns']
    if not turns:
        raise ConversationError('这条追问缺少可用上文，请写出具体的方法名或完整问题。')
    if turns[-1]['claim_count'] != 1 or turns[-1]['answer_truncated']:
        raise ConversationError('上一条回答包含多个要点或未完整保留，请说明你指的是哪个方法或结论。')
    # Only user questions expand retrieval. Previous model claims never become
    # source paragraphs; both generator and verifier receive them as untrusted context.
    anchor = next((t['question'] for t in reversed(turns) if not _FOLLOWUP.search(t['question'])), None)
    if anchor is None:
        raise ConversationError('最初的问题已超出记忆范围，请重新写出具体的方法名或完整问题。')
    return deepcopy(turns), query + '\n' + anchor[:500]


class MemoryConversationStore:
    def __init__(self, clock=utcnow):
        self.clock, self.records, self.lock = clock, {}, threading.RLock()

    def cleanup(self):
        with self.lock:
            now = self.clock()
            self.records = {k: v for k, v in self.records.items()
                            if datetime.fromisoformat(v['expires_at']) > now}

    def create(self, owner, paper, version):
        with self.lock:
            self.cleanup()
            if sum(v['owner_id'] == owner for v in self.records.values()) >= MAX_SESSIONS:
                raise ConversationError('活动对话已达上限，请清空旧对话或等待过期。')
            record = new_conversation(owner, paper, version, self.clock())
            self.records[record['conversation_id']] = record
            return deepcopy(record)

    def get(self, ident, owner):
        with self.lock:
            self.cleanup()
            record = self.records.get(ident)
            return deepcopy(record) if record and record['owner_id'] == owner else None

    def delete(self, ident, owner):
        with self.lock:
            if self.get(ident, owner):
                del self.records[ident]

    def append(self, original, query, result):
        with self.lock:
            current = validate(self.get(original['conversation_id'], original['owner_id']),
                               original['paper_id'], original['paper_version'], original['revision'])
            updated = append_turn(current, query, result)
            self.records[current['conversation_id']] = updated
            return deepcopy(updated)


class PostgresConversationStore:
    def __init__(self, repository):
        self.repository = repository

    def cleanup(self):
        with self.repository.connect() as conn:
            conn.execute('DELETE FROM conversations WHERE expires_at <= clock_timestamp()')

    def create(self, owner, paper, version):
        self.cleanup()
        record = new_conversation(owner, paper, version, utcnow())
        with self.repository.connect() as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 8))', (owner,))
            count = conn.execute('SELECT count(*) AS n FROM conversations WHERE owner_id=%s', (owner,)).fetchone()['n']
            if count >= MAX_SESSIONS:
                raise ConversationError('活动对话已达上限，请清空旧对话或等待过期。')
            conn.execute('INSERT INTO conversations(conversation_id,owner_id,paper_id,paper_version,expires_at) '
                         'VALUES (%s,%s,%s,%s,%s)', tuple(record[k] for k in
                         ('conversation_id', 'owner_id', 'paper_id', 'paper_version', 'expires_at')))
        return record

    def get(self, ident, owner):
        self.cleanup()
        with self.repository.connect() as conn:
            row = conn.execute('SELECT * FROM conversations WHERE conversation_id=%s AND owner_id=%s '
                               'AND expires_at > clock_timestamp()', (ident, owner)).fetchone()
        return {**row, 'expires_at': row['expires_at'].astimezone(timezone.utc).isoformat()} if row else None

    def delete(self, ident, owner):
        with self.repository.connect() as conn:
            conn.execute('DELETE FROM conversations WHERE conversation_id=%s AND owner_id=%s', (ident, owner))

    def append(self, original, query, result):
        updated = append_turn(original, query, result)
        with self.repository.connect() as conn:
            row = conn.execute('UPDATE conversations SET turns=%s, revision=%s WHERE conversation_id=%s '
                               'AND owner_id=%s AND revision=%s AND expires_at > clock_timestamp() RETURNING conversation_id',
                               (Jsonb(updated['turns']), updated['revision'], original['conversation_id'],
                                original['owner_id'], original['revision'])).fetchone()
            if not row:
                raise ConversationError('对话已清空、过期或被更新，本轮未写入记忆。')
        return updated
