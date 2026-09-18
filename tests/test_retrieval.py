import math

import pytest

from research_agent.retrieval import BM25Index, tokenize


def paragraph(chunk_id, text, paper_id="p1"):
    return {"chunk_id": chunk_id, "paper_id": paper_id, "text": text}


def test_bm25_score_matches_hand_calculation_and_scope_is_enforced():
    # Equal length removes length normalization; alpha df=2 in a corpus of 3.
    index = BM25Index([paragraph("a", "alpha beta"), paragraph("b", "beta gamma"),
                       paragraph("c", "alpha alpha", "p2")])
    hits = index.search("alpha", "p1")
    assert [hit.paragraph["chunk_id"] for hit in hits] == ["a"]
    assert hits[0].score == pytest.approx(math.log(1 + 1.5 / 2.5))
    assert index.search("alpha", "p2")[0].paragraph["chunk_id"] == "c"


def test_ties_no_match_and_repeated_query_terms_are_deterministic():
    index = BM25Index([paragraph("b", "alpha beta"), paragraph("a", "alpha beta")])
    assert [h.paragraph["chunk_id"] for h in index.search("alpha", "p1")] == ["a", "b"]
    assert index.search("alpha alpha", "p1") == index.search("alpha", "p1")
    assert index.search("absent", "p1") == []
    assert index.search("alpha", "missing") == []
    assert BM25Index([]).search("alpha", "p1") == []


def test_scientific_tokens_and_invalid_configuration():
    assert tokenize("BERT-base F1 0.91 Table_2") == ["bert-base", "f1", "0.91", "table_2"]
    with pytest.raises(ValueError, match="Duplicate"):
        BM25Index([paragraph("a", "x"), paragraph("a", "y")])
    with pytest.raises(ValueError):
        BM25Index([], b=2)
    with pytest.raises(ValueError):
        BM25Index([]).search("x", "p1", top_k=0)
