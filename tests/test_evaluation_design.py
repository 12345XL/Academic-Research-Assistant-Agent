import copy
import json

import pytest

from research_agent.evaluation_design import (build_manifest, classify, freeze_manifest,
                                              paper_partition, CATEGORIES)


def question(pid, kind, split="train", suffix=""):
    annotation = {"unanswerable": kind == "unanswerable", "yes_no": True if kind == "boolean" else None,
                  "extractive_spans": ["gold secret"] if kind == "extractive" else [],
                  "free_form_answer": "gold secret" if kind == "abstractive" else "",
                  "matched_chunk_ids": [] if kind == "unanswerable" else [pid + ":p0"],
                  "evidence_mappings": [], "figure_evidence": [], "unmapped_evidence": []}
    return {"question_id": f"{pid}-{kind}-{suffix}", "paper_id": pid, "split": split,
            "question": "sensitive question text", "annotations": [annotation]}


def corpus():
    # Two papers in each train pool allow replacing the known diagnostic question.
    pids = {}
    for pool in ("train_tuning", "train_regression"):
        pids[pool] = [f"p{i}" for i in range(100) if paper_partition("train", f"p{i}") == pool][:2]
    return [question(pid, category) for ids in pids.values() for pid in ids for category in CATEGORIES] + [
        question("dev1", category, "dev") for category in CATEGORIES]


def test_balanced_sampling_deterministic_and_paper_disjoint_without_text():
    rows = corpus()
    manifest = build_manifest(rows, {"questions.jsonl": "sha"}, set(), per_category=1)
    assert manifest == build_manifest(list(reversed(rows)), {"questions.jsonl": "sha"}, set(), per_category=1)
    sets = []
    for pool in manifest["pools"].values():
        assert len(pool["cases"]) == 4
        assert set(pool["selected_by_category"].values()) == {1}
        sets.append({q["paper_id"] for q in pool["cases"]})
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert "gold secret" not in json.dumps(manifest)
    assert "sensitive question text" not in json.dumps(manifest)


def test_conflicting_and_figure_labels_get_explicit_exclusions():
    q = question("p", "boolean")
    q["annotations"].append({**q["annotations"][0], "yes_no": False})
    assert classify(q) == (None, "boolean_disagreement")
    q["annotations"][1]["unanswerable"] = True
    assert classify(q) == (None, "answerability_disagreement")
    q = question("p", "extractive")
    q["annotations"][0]["figure_evidence"] = ["figure"]
    assert classify(q) == (None, "figure_evidence")
    q["annotations"][0]["figure_evidence"] = []
    q["annotations"][0]["yes_no"] = True
    assert classify(q) == (None, "inconsistent_answer_fields")


def test_missing_evidence_not_removed_from_generation_denominator():
    rows = corpus()
    for q in rows:
        q["annotations"][0]["matched_chunk_ids"] = []
    result = build_manifest(rows, {}, set(), 1)
    assert all(len(p["cases"]) == 4 for p in result["pools"].values())
    assert not any(c["retrieval_metric_eligible"] for p in result["pools"].values() for c in p["cases"])


def test_known_diagnostic_is_separate_from_selected_sets():
    rows = corpus()
    known_id = rows[0]["question_id"]
    result = build_manifest(rows, {}, {known_id}, 1)
    assert result["known_diagnostics"][0]["question_id"] == known_id
    assert all(c["question_id"] != known_id for p in result["pools"].values() for c in p["cases"])


@pytest.mark.parametrize("fault", ["test", "duplicate", "overlap", "small", "missing_known"])
def test_invalid_corpus_or_quota_rejected(fault):
    rows = corpus(); known = set(); quota = 1
    if fault == "test": rows[0]["split"] = "test"
    if fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if fault == "overlap": rows.append(question(rows[0]["paper_id"], "extractive", "dev", "new"))
    if fault == "small": quota = 100
    if fault == "missing_known": known = {"not-present"}
    with pytest.raises(ValueError): build_manifest(rows, {}, known, quota)


def test_frozen_file_cannot_be_overwritten_after_changing_selection(tmp_path):
    path = tmp_path / "freeze.json"
    assert freeze_manifest(path, {"a": 1}) == "created"
    assert freeze_manifest(path, {"a": 1}) == "unchanged"
    with pytest.raises(ValueError, match="Frozen manifest changed"):
        freeze_manifest(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
