"""Reproducible QASPER v0.3 ingestion; QA annotations never enter the corpus."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
ARCHIVE_NAME = "qasper-train-dev-v0.3.tgz"
# Observed from the official URL on 2026-09-18, not a publisher-signed checksum.
ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
MEMBERS = {"train": "qasper-train-v0.3.json", "dev": "qasper-dev-v0.3.json"}
VERSION = "qasper-v0.3"
ANSWER_FIELDS = (
    "unanswerable", "yes_no", "extractive_spans", "free_form_answer",
    "evidence", "highlighted_evidence",
)
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 80 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{number}: expected an object")
            rows.append(value)
    return rows


def load_papers(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_paragraphs(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def download_archive(raw_dir: Path) -> Path:
    """Use the fixed official train/dev URL and verify cached and new downloads."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / ARCHIVE_NAME
    if not archive_path.exists():
        temporary = archive_path.with_suffix(".tgz.part")
        try:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "research-agent-data/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Archive exceeds download size limit")
                    output.write(block)
            if sha256_file(temporary) != ARCHIVE_SHA256:
                raise ValueError("Downloaded QASPER archive checksum does not match the pinned version")
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Cached QASPER archive checksum mismatch; inspect it before replacing it")
    return archive_path


def read_archive(archive_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Read only allowlisted regular members into memory, never extract paths."""
    datasets = {}
    manifest = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        entries = archive.getmembers()
        for split, name in MEMBERS.items():
            matches = [member for member in entries if member.name == name]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError(f"Expected exactly one regular archive member: {name}")
            member = matches[0]
            if member.size > MAX_MEMBER_BYTES:
                raise ValueError(f"Archive member exceeds size limit: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read archive member: {name}")
            with stream:
                content = stream.read(MAX_MEMBER_BYTES + 1)
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a paper-id keyed object in {name}")
            datasets[split] = data
            manifest[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    overlap = set(datasets["train"]) & set(datasets["dev"])
    if overlap:
        raise ValueError(f"Paper IDs overlap across train/dev: {sorted(overlap)[:5]}")
    return datasets, manifest


def _list_field(value: Any, name: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {name}")
    return value


def _text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Expected text for {name}")
    return value


def convert_split(data: dict[str, dict], split: str) -> dict[str, Any]:
    """Preserve multiple references and explicit labels; map only real paragraphs.

    Whitespace is normalized for matching only. Source text is stored verbatim.
    Missing answer fields remain null and are listed; null is not False.
    """
    if split not in MEMBERS:
        raise ValueError("Only train/dev ingestion is enabled; test remains held out")
    papers, paragraphs, questions = [], [], []
    counts: Counter = Counter()
    seen_question_ids: set[str] = set()
    for paper_id, raw in sorted(data.items()):
        if not isinstance(raw, dict) or not paper_id:
            raise ValueError("Each paper needs a nonempty ID and an object")
        counts["missing_paper_fields"] += sum(field not in raw for field in ("title", "abstract", "full_text", "qas"))
        base = {"paper_id": paper_id, "title": _text(raw.get("title"), "title"),
                "split": split, "source": "qasper", "version": VERSION}
        abstract = _text(raw.get("abstract"), "abstract")
        papers.append({**base, "abstract": abstract})
        paper_paragraphs = []
        if abstract.strip():
            paper_paragraphs.append({**base, "chunk_id": f"{paper_id}:abstract",
                                     "section_name": "Abstract", "section_index": -1,
                                     "paragraph_index": 0, "text": abstract})
            counts["abstract_paragraphs"] += 1
        for section_index, section in enumerate(_list_field(raw.get("full_text"), "full_text")):
            if not isinstance(section, dict):
                raise ValueError(f"Expected section object in {paper_id}")
            section_name = _text(section.get("section_name"), "section_name")
            for paragraph_index, value in enumerate(_list_field(section.get("paragraphs"), "paragraphs")):
                text = _text(value, "paragraph")
                if not text.strip():
                    counts["empty_body_paragraphs_skipped"] += 1
                    continue
                paper_paragraphs.append({**base, "chunk_id": f"{paper_id}:s{section_index}:p{paragraph_index}",
                                         "section_name": section_name, "section_index": section_index,
                                         "paragraph_index": paragraph_index, "text": text})
                counts["body_paragraphs"] += 1
        paragraphs.extend(paper_paragraphs)
        lookup: dict[str, list[str]] = defaultdict(list)
        for paragraph in paper_paragraphs:
            paragraph["text_sha256"] = hashlib.sha256(paragraph["text"].encode("utf-8")).hexdigest()
            lookup[" ".join(paragraph["text"].split())].append(paragraph["chunk_id"])
        for qa in _list_field(raw.get("qas"), "qas"):
            if not isinstance(qa, dict):
                raise ValueError(f"Expected QA object in {paper_id}")
            question_id = qa.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"Missing question_id in {paper_id}")
            if question_id in seen_question_ids:
                raise ValueError(f"Duplicate question_id: {question_id}")
            seen_question_ids.add(question_id)
            question = _text(qa.get("question"), "question")
            if not question.strip():
                raise ValueError(f"Empty question: {question_id}")
            annotations = []
            for reference in _list_field(qa.get("answers"), "answers"):
                if not isinstance(reference, dict) or not isinstance(reference.get("answer", {}), dict):
                    raise ValueError(f"Expected answer object in {question_id}")
                answer = reference.get("answer", {})
                annotation = {field: answer.get(field) for field in ANSWER_FIELDS}
                annotation["annotation_id"] = reference.get("annotation_id")
                annotation["missing_answer_fields"] = [field for field in ANSWER_FIELDS if field not in answer]
                counts["annotations_missing_fields"] += bool(annotation["missing_answer_fields"])
                evidence = _list_field(answer.get("evidence"), "evidence")
                matched, unmapped, figures = [], [], []
                mappings = []
                for item in evidence:
                    text = _text(item, "evidence item")
                    if text.lstrip().startswith("FLOAT SELECTED"):
                        figures.append(item)
                        counts["figure_evidence_items"] += 1
                        continue
                    if not text.strip():
                        counts["empty_evidence_items"] += 1
                        unmapped.append(item)
                        continue
                    ids = lookup.get(" ".join(text.split()), [])
                    if ids:
                        matched.extend(ids)
                        mappings.append({"evidence": item, "chunk_ids": ids})
                        counts["mapped_text_evidence_items"] += 1
                        counts["ambiguous_text_evidence_items"] += len(ids) > 1
                    else:
                        unmapped.append(item)
                        counts["unmapped_text_evidence_items"] += 1
                annotation.update(matched_chunk_ids=list(dict.fromkeys(matched)),
                                  unmapped_evidence=unmapped, figure_evidence=figures,
                                  evidence_mappings=mappings)
                annotations.append(annotation)
                counts["annotations"] += 1
                counts["unanswerable_annotations"] += answer.get("unanswerable") is True
                counts["answerable_annotations"] += answer.get("unanswerable") is False
                counts["unknown_answerability_annotations"] += answer.get("unanswerable") not in (True, False)
                counts["annotations_with_empty_evidence"] += not evidence
                counts["answerable_annotations_with_empty_evidence"] += answer.get("unanswerable") is False and not evidence
                counts["annotations_with_figure_evidence"] += bool(figures)
                counts["annotations_with_unmapped_evidence"] += bool(unmapped)
            labels = {a["unanswerable"] for a in annotations if isinstance(a["unanswerable"], bool)}
            counts["questions_with_answerability_disagreement"] += len(labels) > 1
            counts["questions_with_multiple_annotations"] += len(annotations) > 1
            counts["questions_without_annotations"] += not annotations
            counts["questions_with_figure_evidence"] += any(a["figure_evidence"] for a in annotations)
            counts["questions_with_unmapped_evidence"] += any(a["unmapped_evidence"] for a in annotations)
            counts["questions_with_mapped_text_evidence"] += any(a["matched_chunk_ids"] for a in annotations)
            questions.append({"question_id": question_id, "paper_id": paper_id, "question": question,
                              "split": split, "annotations": annotations})
    counts.update(papers=len(papers), paragraphs=len(paragraphs), questions=len(questions))
    return {"papers": papers, "paragraphs": paragraphs, "questions": questions, "audit": dict(sorted(counts.items()))}


def prepare_dataset(root: Path) -> dict[str, Any]:
    """Download train/dev, convert, and record source and output hashes."""
    archive_path = download_archive(root / "data" / "raw")
    datasets, member_manifest = read_archive(archive_path)
    converted = {split: convert_split(data, split) for split, data in datasets.items()}
    question_ids = [{row["question_id"] for row in converted[split]["questions"]} for split in MEMBERS]
    if question_ids[0] & question_ids[1]:
        raise ValueError("Question IDs overlap across train/dev")
    output_manifest = {}
    for kind in ("papers", "paragraphs", "questions"):
        rows = [row for split in MEMBERS for row in converted[split][kind]]
        output = root / "data" / "processed" / f"{kind}.jsonl"
        _write_jsonl(output, rows)
        output_manifest[f"data/processed/{kind}.jsonl"] = {"rows": len(rows), "sha256": sha256_file(output)}
    audit = {
        "dataset": "QASPER", "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL, "archive_sha256": sha256_file(archive_path),
        "checksum_provenance": "Observed official download on 2026-09-18; pinned locally, not publisher-signed",
        "archive_members": member_manifest,
        "splits": {split: converted[split]["audit"] for split in MEMBERS},
        "integrity": {"train_dev_paper_id_overlap": 0, "train_dev_question_id_overlap": 0,
                      "test_downloaded": False, "test_answers_accessed": False,
                      "qa_annotations_in_retrieval_corpus": False},
        "outputs": output_manifest,
        "limitations": [
            "Text paragraphs and abstracts only; figure/table images are not downloaded or interpreted.",
            "Unmapped or empty evidence is retained, never converted into an unanswerable label.",
            "Multiple annotations, conflicting answerability and duplicate paragraph mappings are retained.",
            "Questions and gold annotations are offline evaluation inputs; never use them as retrieval documents.",
            "Source is structured QASPER text, not a PDF parsing evaluation.",
        ],
    }
    _write_json(root / "data" / "raw" / "manifest.json", {
        "version": VERSION, "source_url": SOURCE_URL, "archive": ARCHIVE_NAME,
        "sha256": audit["archive_sha256"], "members": member_manifest,
        "checksum_provenance": audit["checksum_provenance"],
    })
    _write_json(root / "reports" / "data_audit.json", audit)
    return audit
