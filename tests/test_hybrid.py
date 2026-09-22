import math

import pytest

from research_agent.embeddings import token_windows, vector_literal
from research_agent.retrieval import reciprocal_rank_fusion


def test_rrf_combines_agreement_without_adding_incompatible_scores():
    result = reciprocal_rank_fusion([["lexical", "shared"], ["semantic", "shared"]], 3)
    assert result[0][0] == "shared"
    assert result[0][1] == pytest.approx(2 / 62)
    assert result[1][0] == "lexical"  # deterministic tie break


def test_rrf_deduplicates_a_branch_and_accepts_empty_branch():
    assert reciprocal_rank_fusion([["a", "a", "b"], []], 5) == [("a", 1 / 61), ("b", 1 / 62)]


@pytest.mark.parametrize("length", [0, 1, 510, 511, 956, 2000])
def test_windows_cover_tail_and_respect_model_budget(length):
    tokens = list(range(length))
    windows = token_windows(tokens)
    assert max(map(len, windows)) <= 510
    assert set(token for window in windows for token in window) == set(tokens)
    if len(windows) > 1:
        assert windows[0][-64:] == windows[1][:64]
    assert windows[-1][-1:] == tokens[-1:]


@pytest.mark.parametrize("vector", [[0.0] * 384, [1.0] * 383, [math.nan] * 384, [math.inf] * 384])
def test_invalid_vectors_rejected(vector):
    with pytest.raises(ValueError):
        vector_literal(vector)
