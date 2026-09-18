"""Offline paragraph retrieval evaluation; never a QASPER answer-quality score."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .dataset import read_jsonl, sha256_file
from .retrieval import BM25Index


def gold_references(question: dict) -> tuple[list[set[str]], str | None]:
    """Only unambiguous, completely mapped textual references are eligible.

    Questions with answerability disagreement are excluded. If a question has
    several answerable references, any complete text reference may be used;
    this subset is not representative of all QASPER questions.
    """
    annotations = question["annotations"]
    if not annotations:
        return [], "no_annotations"
    if any(not isinstance(a.get("unanswerable"), bool) for a in annotations):
        return [], "unknown_answerability"
    labels = {a["unanswerable"] for a in annotations}
    if len(labels) > 1:
        return [], "answerability_disagreement"
    if labels == {True}:
        return [], "unanswerable"
    references = []
    for annotation in annotations:
        ids = annotation.get("matched_chunk_ids", [])
        ambiguous = any(len(m["chunk_ids"]) != 1 for m in annotation.get("evidence_mappings", []))
        if ids and not annotation.get("unmapped_evidence") and not annotation.get("figure_evidence") and not ambiguous:
            references.append(set(ids))
    return (references, None) if references else ([], "no_complete_unambiguous_text_reference")


def retrieval_metrics(ranked: list[str], references: list[set[str]]) -> dict[str, float]:
    """Per-question maximum across references, separately for each metric."""
    if not references or any(not reference for reference in references):
        raise ValueError("Metrics need nonempty gold references")
    hits = set(ranked)
    recall = max(len(hits & gold) / len(gold) for gold in references)
    first_ranks = [rank for rank, chunk_id in enumerate(ranked, 1)
                   if any(chunk_id in gold for gold in references)]
    return {"hit": float(bool(first_ranks)), "recall": recall,
            "mrr": 1.0 / first_ranks[0] if first_ranks else 0.0}


def evaluate(data_dir: Path, split: str = "dev", top_k: int = 5, limit: int | None = None) -> dict:
    if split not in {"train", "dev"}:
        raise ValueError("Only train/dev allowed; test is held out")
    if not 1 <= top_k <= 50 or (limit is not None and limit < 1):
        raise ValueError("Invalid top_k or limit")
    paragraphs = read_jsonl(data_dir / "paragraphs.jsonl")
    index = BM25Index(paragraphs)
    questions = sorted((q for q in read_jsonl(data_dir / "questions.jsonl") if q["split"] == split),
                       key=lambda q: q["question_id"])
    exclusions: Counter = Counter()
    eligible = []
    for question in questions:
        references, reason = gold_references(question)
        if reason:
            exclusions[reason] += 1
        else:
            for reference in references:
                if any(cid not in index.paragraphs or index.paragraphs[cid]["paper_id"] != question["paper_id"]
                       for cid in reference):
                    raise ValueError("Gold evidence refers to a missing paragraph or another paper")
            eligible.append((question, references))
    selected = eligible if limit is None else eligible[:limit]
    cases = []
    for question, references in selected:
        ranked = [hit.paragraph["chunk_id"] for hit in index.search(question["question"], question["paper_id"], top_k)]
        cases.append({"question_id": question["question_id"], "paper_id": question["paper_id"],
                      "gold_reference_count": len(references), "retrieved_chunk_ids": ranked,
                      **retrieval_metrics(ranked, references)})
    metrics = {f"{metric}_at_{top_k}": sum(case[metric] for case in cases) / len(cases) if cases else None
               for metric in ("hit", "recall", "mrr")}
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_hash = hashlib.sha256()
    for path in source_files:
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    return {
        "protocol": "specified-paper-text-evidence-bm25-v1", "stage": "P1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "app_version": __version__,
        "source_sha256": source_hash.hexdigest(), "split": split,
        "config": {"top_k": top_k, "k1": index.k1, "b": index.b,
                   "corpus": "train+dev abstracts and body paragraphs; global IDF; candidates restricted to known paper",
                   "selection": "all eligible sorted by question_id" if limit is None else f"first {limit} eligible sorted by question_id",
                   "reference_policy": "maximum per metric across complete unambiguous textual answerable references",
                   "model_calls": 0},
        "counts": {"split_questions": len(questions), "eligible_questions": len(eligible),
                   "evaluated_questions": len(cases), "exclusions": dict(sorted(exclusions.items()))},
        "data_sha256": {name: sha256_file(data_dir / name) for name in ("paragraphs.jsonl", "questions.jsonl")},
        "metrics": metrics,
        "limitations": ["Not official QASPER answer evaluation; no generated answer, verifier, or refusal evaluation",
                        "Known-paper retrieval; not corpus-wide paper discovery or cross-paper reasoning",
                        "Text-reference subset excludes unanswerability/disagreement and cases lacking a complete text reference",
                        "Abstracts included; figures/tables and PDF parsing are not covered; test split remains unused"],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("reports/p1_retrieval_dev.json"))
    args = parser.parse_args()
    report = evaluate(args.data_dir, args.split, args.top_k, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol", "counts", "metrics", "limitations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
