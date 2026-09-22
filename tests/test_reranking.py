import math

import pytest

from research_agent.reranking import LocalReranker, RerankerUnavailableError, apply_reranking, pair_windows
from research_agent.service import CorpusChangedError, CorpusIntegrityError, PersistentEvidenceService
from test_persistent_service import RepositoryDouble


def test_pair_windows_cover_tail_and_segment_ids_without_query_truncation():
    query, passage = list(range(1000, 1128)), list(range(1500))
    windows = list(pair_windows(query, passage, -1, -2))
    assert len(windows) > 1
    covered = set()
    for ids, types in windows:
        assert len(ids) == len(types) <= 512
        assert ids[:130] == [-1, *query, -2]
        assert types[:130] == [0] * 130
        assert types[130:] == [1] * (len(ids) - 130)
        covered.update(ids[130:-1])
    assert covered == set(passage)


@pytest.mark.parametrize("size", [0, 129])
def test_query_budget_rejects_instead_of_silent_truncation(size):
    with pytest.raises(ValueError):
        list(pair_windows([1] * size, [2], 101, 102))


def test_negative_logits_and_ties_keep_real_candidate_identity():
    candidates = [{"chunk_id": key, "rank": rank, "score": 10 - rank, "text": key}
                  for rank, key in enumerate(["b", "a", "c"], 1)]
    result = apply_reranking(candidates, [-2, -2, -1], 2)
    assert [r["chunk_id"] for r in result] == ["c", "a"]
    assert result[0]["candidate_rank"] == 3 and result[0]["score"] == -1
    assert result[0]["text"] == "c"
    assert candidates[0]["rank"] == 1  # do not mutate upstream ranking


@pytest.mark.parametrize("scores", [[], [math.nan], [math.inf]])
def test_invalid_scores_fail_closed(scores):
    with pytest.raises(RerankerUnavailableError):
        apply_reranking([{"chunk_id": "a", "rank": 1, "score": 2}], scores, 5)


def test_missing_local_model_does_not_download(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_RERANKER_DIR", str(tmp_path))
    with pytest.raises(RerankerUnavailableError):
        LocalReranker().score("query", ["text"])


def test_window_budget_prevents_unbounded_model_work():
    class Tokenizer:
        cls_token_id, sep_token_id = 101, 102
        def __call__(self, value, **kwargs):
            return {"input_ids": [1] if isinstance(value, str) else [[2] * 150000]}
    reranker = LocalReranker()
    reranker._model, reranker._tokenizer = object(), Tokenizer()
    with pytest.raises(ValueError, match="256"):
        reranker.score("query", ["very long paragraph"])


def test_fact_check_precedes_reranker_and_unknown_paper_never_calls_model():
    class Reranker:
        def score(self, query, texts):
            pytest.fail("Unverified facts must not reach the model")
    repo = RepositoryDouble()
    service = PersistentEvidenceService(repo, reranker=Reranker())
    service.retrieve("alpha", "p1")
    repo.paragraph["text"] = "tampered"
    with pytest.raises(CorpusIntegrityError):
        service.retrieve("alpha", "p1", rerank=True)
    with pytest.raises(KeyError):
        service.retrieve("alpha", "missing", rerank=True)


def test_version_change_during_reranking_retries_all_candidates():
    repo = RepositoryDouble()
    class Reranker:
        calls = 0
        def score(self, query, texts):
            self.calls += 1
            if self.calls == 1:
                repo.change_text("alpha replacement")
                repo.current_revision += 1
            return [1.0] * len(texts), {"windows": len(texts)}
    reranker = Reranker()
    result = PersistentEvidenceService(repo, reranker=reranker).retrieve("alpha", "p1", rerank=True)
    assert reranker.calls == 2
    assert result["citations"][0]["text"] == "alpha replacement"
    assert result["trace"]["corpus_revision"] == "2"


def test_repeated_version_change_terminates_without_stale_result():
    repo = RepositoryDouble()
    class Reranker:
        def score(self, query, texts):
            repo.current_revision += 1
            return [1.0] * len(texts), {}
    with pytest.raises(CorpusChangedError):
        PersistentEvidenceService(repo, reranker=Reranker()).retrieve("alpha", "p1", rerank=True)


def test_empty_candidates_do_not_require_model():
    result = PersistentEvidenceService(RepositoryDouble()).retrieve("unknownterm", "p1", rerank=True)
    assert result["citations"] == []
    assert result["trace"]["model_calls"] == 0
    assert result["trace"]["reranker_stats"]["windows"] == 0
