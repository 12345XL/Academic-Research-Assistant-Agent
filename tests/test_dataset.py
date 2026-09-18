"""Offline checks for provenance, corpus isolation and evidence label semantics."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from research_agent.dataset import (
    MEMBERS, convert_split, load_papers, load_paragraphs, prepare_dataset, read_archive,
)


def answer(**overrides):
    return {"answer": {"unanswerable": False, "yes_no": None, "extractive_spans": [],
                       "free_form_answer": "GOLD_ANSWER_ONLY", "evidence": ["A body paragraph."],
                       "highlighted_evidence": ["A body paragraph."], **overrides},
            "annotation_id": "a1"}


def paper():
    return {"title": "A research paper", "abstract": "An abstract.",
            "full_text": [{"section_name": "Method", "paragraphs": ["", "A body paragraph."]}],
            "qas": [{"question_id": "q1", "question": "QUESTION_ONLY: What was used?", "answers": [answer()]}]}


def write_archive(path, train, dev, unsafe_member=None):
    with tarfile.open(path, "w:gz") as archive:
        for split, data in (("train", train), ("dev", dev)):
            content = json.dumps(data).encode()
            member = tarfile.TarInfo(MEMBERS[split])
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if unsafe_member:
            member = tarfile.TarInfo(unsafe_member)
            member.size = 4
            archive.addfile(member, io.BytesIO(b"evil"))


class DatasetTests(unittest.TestCase):
    def test_gold_questions_and_answers_never_become_corpus(self):
        converted = convert_split({"p1": paper()}, "train")
        corpus = json.dumps([converted["papers"], converted["paragraphs"]])
        self.assertNotIn("GOLD_ANSWER_ONLY", corpus)
        self.assertNotIn("QUESTION_ONLY", corpus)
        self.assertEqual(converted["questions"][0]["annotations"][0]["free_form_answer"], "GOLD_ANSWER_ONLY")

    def test_empty_paragraph_does_not_shift_source_location(self):
        converted = convert_split({"p1": paper()}, "train")
        paragraph = converted["paragraphs"][1]
        self.assertEqual((paragraph["chunk_id"], paragraph["section_index"], paragraph["paragraph_index"]),
                         ("p1:s0:p1", 0, 1))
        self.assertEqual(converted["questions"][0]["annotations"][0]["matched_chunk_ids"], ["p1:s0:p1"])

    def test_empty_evidence_does_not_imply_unanswerable(self):
        raw = paper()
        raw["qas"][0]["answers"] = [answer(evidence=[], yes_no=False, free_form_answer="")]
        converted = convert_split({"p1": raw}, "train")
        annotation = converted["questions"][0]["annotations"][0]
        self.assertIs(annotation["unanswerable"], False)
        self.assertIs(annotation["yes_no"], False)
        self.assertEqual(annotation["matched_chunk_ids"], [])
        self.assertEqual(converted["audit"]["answerable_annotations_with_empty_evidence"], 1)

    def test_text_hash_uses_exact_utf8_source_without_normalizing_whitespace(self):
        raw = paper()
        source_text = "  原始段落: A  body\nparagraph.\t"
        raw["full_text"][0]["paragraphs"][1] = source_text
        raw["qas"][0]["answers"][0]["answer"]["evidence"] = ["原始段落: A body paragraph."]
        converted = convert_split({"p1": raw}, "train")
        paragraph = converted["paragraphs"][1]
        self.assertEqual(paragraph["text"], source_text)
        self.assertEqual(paragraph["text_sha256"], hashlib.sha256(source_text.encode("utf-8")).hexdigest())
        self.assertNotEqual(paragraph["text_sha256"], hashlib.sha256(" ".join(source_text.split()).encode("utf-8")).hexdigest())
        self.assertEqual(converted["questions"][0]["annotations"][0]["matched_chunk_ids"], ["p1:s0:p1"])
        self.assertEqual(converted["paragraphs"][0]["text_sha256"], hashlib.sha256(raw["abstract"].encode("utf-8")).hexdigest())

    def test_multiple_annotations_and_disagreement_are_preserved(self):
        raw = paper()
        raw["qas"][0]["answers"].append(answer(unanswerable=True, evidence=[], free_form_answer=""))
        converted = convert_split({"p1": raw}, "dev")
        self.assertEqual([a["unanswerable"] for a in converted["questions"][0]["annotations"]], [False, True])
        self.assertEqual(converted["audit"]["questions_with_answerability_disagreement"], 1)

    def test_missing_fields_remain_unknown_and_are_audited(self):
        raw = paper()
        raw.pop("abstract")
        raw["qas"][0]["answers"] = [{"answer": {"evidence": []}}]
        converted = convert_split({"p1": raw}, "train")
        annotation = converted["questions"][0]["annotations"][0]
        self.assertIsNone(annotation["unanswerable"])
        self.assertIn("unanswerable", annotation["missing_answer_fields"])
        self.assertEqual(converted["audit"]["unknown_answerability_annotations"], 1)
        self.assertEqual(converted["audit"]["missing_paper_fields"], 1)

    def test_figures_unmapped_evidence_and_whitespace_are_distinct(self):
        raw = paper()
        raw["qas"][0]["answers"] = [answer(evidence=["A  body\nparagraph.", "FLOAT SELECTED: Table 1", "Not in the paper"])]
        original = copy.deepcopy(raw)
        annotation = convert_split({"p1": raw}, "train")["questions"][0]["annotations"][0]
        self.assertEqual(annotation["matched_chunk_ids"], ["p1:s0:p1"])
        self.assertEqual(annotation["figure_evidence"], ["FLOAT SELECTED: Table 1"])
        self.assertEqual(annotation["unmapped_evidence"], ["Not in the paper"])
        self.assertEqual(annotation["evidence"], original["qas"][0]["answers"][0]["answer"]["evidence"])
        self.assertEqual(raw, original)

    def test_duplicate_source_text_retains_both_locations(self):
        raw = paper()
        raw["full_text"][0]["paragraphs"].append("A body paragraph.")
        converted = convert_split({"p1": raw}, "train")
        self.assertEqual(converted["questions"][0]["annotations"][0]["matched_chunk_ids"], ["p1:s0:p1", "p1:s0:p2"])
        self.assertEqual(converted["audit"]["ambiguous_text_evidence_items"], 1)

    def test_bad_identity_or_unsupported_split_fails(self):
        raw = paper()
        raw["qas"][0].pop("question_id")
        with self.assertRaisesRegex(ValueError, "question_id"):
            convert_split({"p1": raw}, "train")
        with self.assertRaisesRegex(ValueError, "test remains held out"):
            convert_split({"p1": paper()}, "test")

    def test_archive_does_not_extract_untrusted_paths_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.tgz"
            write_archive(path, {"p1": paper()}, {}, unsafe_member="../SHOULD_NOT_EXIST")
            data, manifest = read_archive(path)
            self.assertIn("p1", data["train"])
            self.assertEqual(set(manifest), set(MEMBERS.values()))
            self.assertEqual(list(root.iterdir()), [path])
            write_archive(path, {"p1": paper()}, {"p1": paper()})
            with self.assertRaisesRegex(ValueError, "overlap"):
                read_archive(path)

    def test_archive_rejects_symlink_for_expected_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tgz"
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo(MEMBERS["train"])
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, "regular archive member"):
                read_archive(path)

    def test_prepare_records_hashes_and_separates_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.tgz"
            write_archive(archive, {"p1": paper()}, {})
            with patch("research_agent.dataset.download_archive", return_value=archive):
                audit = prepare_dataset(root)
            self.assertFalse(audit["integrity"]["test_answers_accessed"])
            self.assertEqual(audit["outputs"]["data/processed/questions.jsonl"]["rows"], 1)
            self.assertEqual(len(load_papers(root / "data/processed/papers.jsonl")), 1)
            self.assertEqual(len(load_paragraphs(root / "data/processed/paragraphs.jsonl")), 2)
            self.assertTrue((root / "data/raw/manifest.json").is_file())
            self.assertTrue((root / "reports/data_audit.json").is_file())


if __name__ == "__main__":
    unittest.main()
