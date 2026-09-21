"""Corpus publication preconditions without a running PostgreSQL/S3 service."""

import copy
import hashlib
import json
from unittest.mock import Mock

import pytest

from research_agent import ingestion
from research_agent.ingestion import CorpusValidationError, build_documents, ingest_qasper


def paper(paper_id="p1", **changes):
    return {"paper_id": paper_id, "title": f"Paper {paper_id}", "abstract": "Original abstract",
            "split": "train", "source": "qasper", "version": "qasper-v0.3", **changes}


def paragraph(parent, *, text="Original evidence", section=0, index=0, **changes):
    return {key: value for key, value in {
        **parent, "chunk_id": f"{parent['paper_id']}:s{section}:p{index}",
        "section_name": "Methods", "section_index": section, "paragraph_index": index,
        "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        **changes,
    }.items() if key != "abstract"}


@pytest.mark.parametrize("field", ["qas", "questions", "answers", "annotations", "evidence", "unanswerable", "gold"])
@pytest.mark.parametrize("record", ["paper", "paragraph"])
def test_gold_fields_are_rejected_instead_of_silently_removed(field, record):
    parent = paper()
    chunk = paragraph(parent)
    (parent if record == "paper" else chunk)[field] = "PRIVATE_GOLD_LABEL"
    with pytest.raises(CorpusValidationError, match="Evaluation labels are forbidden"):
        build_documents([parent], [chunk])


@pytest.mark.parametrize("record", ["paper", "paragraph"])
def test_allowlist_rejects_unrecognized_fields_including_nested_gold(record):
    parent = paper()
    chunk = paragraph(parent)
    (parent if record == "paper" else chunk)["extra_metadata"] = {"answer": "PRIVATE_GOLD_LABEL"}
    with pytest.raises(CorpusValidationError, match="approved schema"):
        build_documents([parent], [chunk])


def test_canonical_documents_sort_chunks_preserve_source_and_do_not_mutate_input():
    a, b = paper("a"), paper("b")
    exact = "  原始证据  has\nwhitespace.\t"
    chunks = [paragraph(b), paragraph(a, section=1), paragraph(a, text=exact, section=0),
              paragraph(a, text=a["abstract"], section=-1)]
    before = copy.deepcopy(([b, a], chunks))
    first_manifest, first = build_documents([b, a], chunks)
    reordered_papers = [dict(reversed(list(row.items()))) for row in [a, b]]
    second_manifest, second = build_documents(reordered_papers, list(reversed(chunks)))
    assert first_manifest == second_manifest
    assert [d.content for d in first] == [d.content for d in second]
    assert [d.version_id for d in first] == [d.version_id for d in second]
    assert [d.paper["paper_id"] for d in first] == ["a", "b"]
    decoded = json.loads(first[0].content)
    assert [c["section_index"] for c in decoded["paragraphs"]] == [-1, 0, 1]
    assert decoded["paragraphs"][1]["text"] == exact
    assert ([b, a], chunks) == before
    assert first[0].sha256 == hashlib.sha256(first[0].content).hexdigest()
    assert first[0].object_key.endswith(first[0].sha256 + ".json")


def test_content_change_produces_new_immutable_object_version_and_manifest():
    parent = paper()
    first_manifest, first = build_documents([parent], [paragraph(parent)])
    second_manifest, second = build_documents([parent], [paragraph(parent, text="Changed evidence")])
    assert first_manifest != second_manifest
    assert first[0].version_id != second[0].version_id
    assert first[0].object_key != second[0].object_key


@pytest.mark.parametrize("changes,match", [
    ({"text": "Changed without a new checksum"}, "checksum mismatch"),
    ({"paper_id": "unknown"}, "unknown paper"),
    ({"title": "Another paper"}, "metadata does not match"),
    ({"split": "dev"}, "metadata does not match"),
    ({"source": "personal-upload"}, "metadata does not match"),
    ({"version": "qasper-v0.2"}, "metadata does not match"),
    ({"section_index": True}, "section index"),
    ({"paragraph_index": -1}, "nonnegative integer"),
    ({"text": " \n\t"}, "text must be nonempty"),
])
def test_checksum_identity_and_location_mismatches_are_rejected(changes, match):
    parent = paper()
    chunk = paragraph(parent)
    chunk.update(changes)
    with pytest.raises(CorpusValidationError, match=match):
        build_documents([parent], [chunk])


def test_duplicate_ids_or_unsearchable_papers_are_rejected():
    parent = paper()
    chunk = paragraph(parent)
    for papers, paragraphs, match in (
        ([], [], "at least one"),
        ([parent, dict(parent)], [chunk], "Duplicate paper"),
        ([parent], [chunk, dict(chunk)], "unique"),
        ([parent], [], "searchable paragraph"),
    ):
        with pytest.raises(CorpusValidationError, match=match):
            build_documents(papers, paragraphs)


def test_test_split_is_never_accepted_for_development_import():
    parent = paper(split="test")
    with pytest.raises(CorpusValidationError, match="test remains held out"):
        build_documents([parent], [paragraph(parent)])


def test_invalid_corpus_fails_before_database_or_object_side_effects(monkeypatch, tmp_path):
    parent = paper()
    parent["answers"] = ["PRIVATE_GOLD_LABEL"]
    monkeypatch.setattr(ingestion, "load_papers", lambda path: [parent])
    monkeypatch.setattr(ingestion, "load_paragraphs", lambda path: [])
    repo, store = Mock(), Mock()
    with pytest.raises(CorpusValidationError, match="Evaluation labels"):
        ingest_qasper(repo, store, tmp_path)
    assert repo.mock_calls == []
    assert store.mock_calls == []
