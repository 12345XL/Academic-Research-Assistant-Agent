import json

import pytest

from research_agent.evaluate import evaluate, gold_references, retrieval_metrics


def annotation(ids=None, unanswerable=False, **kwargs):
    return {"unanswerable": unanswerable, "matched_chunk_ids": ids or [],
            "unmapped_evidence": [], "figure_evidence": [], "evidence_mappings": [], **kwargs}


def test_hit_recall_and_mrr_mean_different_things():
    result = retrieval_metrics(["wrong", "a"], [{"a", "b"}])
    assert result == {"hit": 1.0, "recall": 0.5, "mrr": 0.5}
    # References stay separate: annotator 2's single sufficient paragraph is complete.
    assert retrieval_metrics(["c"], [{"a", "b"}, {"c"}])["recall"] == 1.0
    assert retrieval_metrics(["wrong"], [{"a"}]) == {"hit": 0.0, "recall": 0.0, "mrr": 0.0}
    with pytest.raises(ValueError):
        retrieval_metrics([], [set()])


def test_answerability_and_evidence_coverage_are_not_conflated():
    assert gold_references({"annotations": [annotation([], yes_no=False)]})[1] == "no_complete_unambiguous_text_reference"
    assert gold_references({"annotations": [annotation([], unanswerable=True)]})[1] == "unanswerable"
    assert gold_references({"annotations": [annotation(["a"]), annotation([], unanswerable=True)]})[1] == "answerability_disagreement"
    assert gold_references({"annotations": [annotation(["a"], figure_evidence=["FLOAT SELECTED: table"])]})[0] == []
    assert gold_references({"annotations": [annotation(["a"], unmapped_evidence=["missing"])]})[0] == []
    assert gold_references({"annotations": [annotation(["a", "b"], evidence_mappings=[{"chunk_ids": ["a", "b"]}])]})[0] == []
    assert gold_references({"annotations": [annotation(["a"]), annotation([], figure_evidence=["FLOAT SELECTED"])]})[0] == [{"a"}]


def test_evaluation_denominators_and_scope_with_known_answers(tmp_path):
    paragraphs = [{"chunk_id": "a", "paper_id": "p1", "text": "alpha evidence"},
                  {"chunk_id": "b", "paper_id": "p2", "text": "alpha alpha"}]
    questions = [{"question_id": "1", "paper_id": "p1", "question": "alpha", "split": "dev", "annotations": [annotation(["a"])]},
                 {"question_id": "2", "paper_id": "p1", "question": "absent", "split": "dev", "annotations": [annotation([], unanswerable=True)]}]
    for name, rows in [("paragraphs", paragraphs), ("questions", questions)]:
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = evaluate(tmp_path)
    assert result["counts"] == {"split_questions": 2, "eligible_questions": 1, "evaluated_questions": 1, "exclusions": {"unanswerable": 1}}
    assert result["metrics"] == {"hit_at_5": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}
    assert result["cases"][0]["retrieved_chunk_ids"] == ["a"]
    with pytest.raises(ValueError, match="test is held out"):
        evaluate(tmp_path, split="test")
