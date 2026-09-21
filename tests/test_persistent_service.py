import hashlib

import pytest

from research_agent.service import CorpusChangedError, CorpusIntegrityError, PersistentEvidenceService


class RepositoryDouble:
    def __init__(self):
        self.current_revision = 1
        self.online = True
        self.loads = 0
        self.paper = {"paper_id": "p1", "title": "Paper", "source": "qasper", "version": "v1"}
        self.change_text("alpha evidence")

    def change_text(self, text):
        self.paragraph = {**self.paper, "chunk_id": "a", "section_name": "Results", "section_index": 0,
                          "paragraph_index": 0, "text": text,
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def revision(self):
        if not self.online:
            raise ConnectionError("Database unavailable")
        return self.current_revision

    def load_snapshot(self):
        self.loads += 1
        return self.current_revision, [dict(self.paper)], [dict(self.paragraph)]

    def get_chunks(self, chunk_ids, paper_id):
        return {"a": dict(self.paragraph)} if chunk_ids and paper_id == "p1" else {}


def test_database_revision_refresh_and_outage_fail_closed():
    repo = RepositoryDouble()
    service = PersistentEvidenceService(repo)
    first = service.retrieve("alpha", "p1")
    assert first["citations"][0]["text"] == "alpha evidence"
    assert first["trace"]["fact_check"] == "database"
    service.retrieve("alpha", "p1")
    assert repo.loads == 1
    repo.change_text("beta replacement")
    repo.current_revision += 1
    assert service.retrieve("alpha", "p1")["citations"] == []
    assert service.retrieve("beta", "p1")["citations"][0]["text"] == "beta replacement"
    repo.online = False
    with pytest.raises(ConnectionError):
        service.retrieve("beta", "p1")


def test_missing_fact_never_returns_stale_cached_text():
    repo = RepositoryDouble()
    repo.get_chunks = lambda chunk_ids, paper_id: {}
    with pytest.raises(CorpusChangedError):
        PersistentEvidenceService(repo).retrieve("alpha", "p1")


def test_tampered_database_text_never_returns_as_verified_evidence():
    repo = RepositoryDouble()
    service = PersistentEvidenceService(repo)
    assert service.retrieve("alpha", "p1")["citations"]
    repo.paragraph["text"] = "altered evidence"
    with pytest.raises(CorpusIntegrityError):
        service.retrieve("alpha", "p1")


def test_unknown_paper_and_no_lexical_match_are_different():
    service = PersistentEvidenceService(RepositoryDouble())
    with pytest.raises(KeyError):
        service.retrieve("alpha", "missing")
    assert service.retrieve("missingterm", "p1")["status"] == "no_lexical_match"
