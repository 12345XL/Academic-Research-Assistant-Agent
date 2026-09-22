"""Offline-only evaluation sampling. No model clients and no online imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .evaluate import gold_references

CATEGORIES = ("extractive", "abstractive", "boolean", "unanswerable")
SEED = "qasper-quality-v2-20260922"


def fingerprint(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest()


def classify(question: dict) -> tuple[str | None, str | None]:
    """Strict agreement subset; all exclusions are retained in the inventory."""
    annotations = question.get("annotations", [])
    if not annotations:
        return None, "missing_annotations"
    if any(type(a.get("unanswerable")) is not bool for a in annotations):
        return None, "unknown_answerability"
    if len({a["unanswerable"] for a in annotations}) != 1:
        return None, "answerability_disagreement"
    if any(a.get("figure_evidence") for a in annotations):
        return None, "figure_evidence"
    if any(a.get("missing_answer_fields") for a in annotations):
        return None, "missing_answer_fields"
    categories = set()
    for a in annotations:
        kinds = []
        if a["unanswerable"]:
            kinds.append("unanswerable")
        if type(a.get("yes_no")) is bool:
            kinds.append("boolean")
        if a.get("extractive_spans"):
            kinds.append("extractive")
        if a.get("free_form_answer", "").strip():
            kinds.append("abstractive")
        if len(kinds) != 1:
            return None, "inconsistent_answer_fields"
        categories.add(kinds[0])
    if len(categories) != 1:
        return None, "mixed_answer_types"
    category = next(iter(categories))
    if category == "boolean" and len({a["yes_no"] for a in annotations}) != 1:
        return None, "boolean_disagreement"
    return category, None


def paper_partition(split: str, paper_id: str) -> str:
    if split == "dev":
        return "dev_comparison"
    # Paper-wise partition: no sibling questions leak between train pools.
    return "train_regression" if int(fingerprint("paper", paper_id), 16) % 5 == 0 else "train_tuning"


def build_manifest(questions: list[dict], data_sha256: dict[str, str], known_case_ids: set[str],
                   per_category: int = 20) -> dict:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    seen_ids, paper_splits = set(), {}
    pool_rows = {p: [] for p in ("train_tuning", "train_regression", "dev_comparison")}
    inventories = {s: Counter() for s in ("train", "dev")}
    exclusions, known_cases = [], []
    for question in sorted(questions, key=lambda q: q["question_id"]):
        split, pid, qid = question["split"], question["paper_id"], question["question_id"]
        if split not in {"train", "dev"}:
            raise ValueError("Only train/dev allowed; official test is held out")
        if qid in seen_ids:
            raise ValueError("Duplicate question ID")
        seen_ids.add(qid)
        if pid in paper_splits and paper_splits[pid] != split:
            raise ValueError("Paper overlaps official splits")
        paper_splits[pid] = split
        category, reason = classify(question)
        inventories[split][category or reason] += 1
        row = {"question_id": qid, "paper_id": pid, "split": split}
        if reason:
            exclusions.append({**row, "reason": reason})
            if qid in known_case_ids:
                known_cases.append({**row, "excluded_reason": reason})
            continue
        references, evidence_reason = gold_references(question)
        row.update(category=category, retrieval_metric_eligible=bool(references),
                   retrieval_exclusion_reason=evidence_reason)
        if qid in known_case_ids:
            known_cases.append(row)
        else:
            pool_rows[paper_partition(split, pid)].append(row)
    if known_case_ids - seen_ids:
        raise ValueError("Known smoke case is missing from data")
    pools = {}
    for name, rows in pool_rows.items():
        selected, available = [], {}
        for category in CATEGORIES:
            candidates = sorted((r for r in rows if r["category"] == category),
                                key=lambda r: (fingerprint("question", r["question_id"]), r["question_id"]))
            available[category] = len(candidates)
            if len(candidates) < per_category:
                raise ValueError(f"Insufficient {name}/{category}: {len(candidates)} < {per_category}")
            selected.extend(candidates[:per_category])
        pools[name] = {"available_by_category": available, "selected_by_category": {
                      c: per_category for c in CATEGORIES}, "cases": selected}
    return {
        "protocol": "qasper-stratified-quality-v2", "seed": SEED,
        "status": "selection_frozen_no_model_evaluation", "per_category_per_pool": per_category,
        "data_sha256": dict(sorted(data_sha256.items())),
        "selection_rules": {
            "train_partition": "SHA256(seed:paper:paper_id) modulo 5; zero -> regression, others -> tuning",
            "within_category": "ascending SHA256(seed:question:question_id), then question_id",
            "strata": list(CATEGORIES), "known_smoke_cases": "separate diagnostic set, not fresh evaluation",
            "dev_warning": "Development comparison only; retrieval dev results have already been inspected",
            "official_test": "not read", "sampling": "balanced quotas; not prevalence-weighted dataset accuracy",
        },
        "inventory": {s: dict(sorted(counts.items())) for s, counts in inventories.items()},
        "pools": pools, "known_diagnostics": known_cases, "excluded_cases": exclusions,
    }


def freeze_manifest(path: Path, manifest: dict) -> str:
    """Never silently replace a protocol or sample after observing model outputs."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Frozen manifest changed; create a reviewed version, do not overwrite")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also avoids overwriting a concurrently frozen manifest.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return "created"
