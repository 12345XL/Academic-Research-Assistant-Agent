"""Frozen six-arm reranker ablation through the actual fact-checked service."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_agent import __version__
from research_agent.dataset import read_jsonl, sha256_file
from research_agent.embeddings import LocalEncoder
from research_agent.evaluate import gold_references, retrieval_metrics
from research_agent.reranking import CONFIG, LocalReranker
from research_agent.service import PersistentEvidenceService
from research_agent.settings import Settings
from research_agent.storage import Repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    previous = json.loads((ROOT / "reports/p2b_retrieval_dev.json").read_text())
    for filename, checksum in previous["data_sha256"].items():
        if sha256_file(ROOT / "data/processed" / filename) != checksum:
            raise ValueError("Frozen evaluation data changed")
    questions = {q["question_id"]: q for q in read_jsonl(ROOT / "data/processed/questions.jsonl")}
    selected = [questions[c["question_id"]] for c in previous["cases"]]
    assert len(selected) == 745 and all(q["split"] == "dev" for q in selected)
    repo = Repository(Settings.from_env())
    revision = repo.revision()
    assert revision == previous["corpus_revision"]
    # CPU query encoder matches the frozen P2B-1 encoder, independently of reranker device.
    encoder = LocalEncoder("cpu")
    started = time.perf_counter()
    vectors = encoder.encode_queries([q["question"] for q in selected])
    embedding_seconds = time.perf_counter() - started
    service = PersistentEvidenceService(repo, encoder=encoder, reranker=LocalReranker(args.device))
    cases, times = [], {mode: [] for mode in ("bm25", "dense", "hybrid")}
    for number, (question, previous_case, vector) in enumerate(zip(selected, previous["cases"], vectors), 1):
        references, reason = gold_references(question)
        assert reason is None
        case = {"question_id": question["question_id"], "paper_id": question["paper_id"]}
        for mode in times:
            baseline = service.retrieve(question["question"], question["paper_id"], 5, mode=mode, query_vector=vector)
            ids = [c["chunk_id"] for c in baseline["citations"]]
            if ids != previous_case[mode]["retrieved_chunk_ids"]:
                raise ValueError(f"Frozen {mode} ranking changed")
            reranked = service.retrieve(question["question"], question["paper_id"], 5,
                                        mode=mode, query_vector=vector, rerank=True)
            assert reranked["trace"]["corpus_revision"] == str(revision)
            candidates = reranked["trace"]["rerank_candidates"]
            actual = [c["chunk_id"] for c in reranked["citations"]]
            assert set(actual) <= set(candidates) and len(candidates) <= 20
            assert ids == candidates[:5]  # same first-stage ordering and candidate budget
            case[mode] = {"retrieved_chunk_ids": ids, **retrieval_metrics(ids, references)}
            case[mode + "_rerank"] = {
                "retrieved_chunk_ids": actual, **retrieval_metrics(actual, references),
                "candidate_chunk_ids": candidates, "candidate_metrics": retrieval_metrics(candidates, references),
                "rerank_latency_ms": reranked["trace"]["rerank_latency_ms"],
                "reranker_stats": reranked["trace"]["reranker_stats"],
            }
            times[mode].append(reranked["trace"]["rerank_latency_ms"])
        cases.append(case)
        if number % 25 == 0:
            print(f"Evaluated {number}/745", flush=True)
    assert repo.revision() == revision
    metrics, differences = {}, {}
    for mode in times:
        for arm in (mode, mode + "_rerank"):
            metrics[arm] = {f"{metric}_at_5": statistics.mean(c[arm][metric] for c in cases)
                            for metric in ("hit", "recall", "mrr")}
        differences[mode] = {
            "new_hits": sum(c[mode + "_rerank"]["hit"] > c[mode]["hit"] for c in cases),
            "lost_hits": sum(c[mode + "_rerank"]["hit"] < c[mode]["hit"] for c in cases),
            "candidate_hit_at_20": statistics.mean(c[mode + "_rerank"]["candidate_metrics"]["hit"] for c in cases),
            "candidate_recall_at_20": statistics.mean(c[mode + "_rerank"]["candidate_metrics"]["recall"] for c in cases),
            "not_recalled_in_candidates": sum(c[mode + "_rerank"]["candidate_metrics"]["hit"] == 0 for c in cases),
            "recalled_but_not_in_reranked_top5": sum(c[mode + "_rerank"]["candidate_metrics"]["hit"] > c[mode + "_rerank"]["hit"] for c in cases),
        }
    report = {
        "stage": "P2B-2", "protocol": "specified-paper-text-evidence-six-arm-rerank-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "app_version": __version__,
        "questions": len(cases), "config": CONFIG, "candidate_pool": 20, "top_k": 5,
        "source_sha256": hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
            for p in sorted((ROOT / "src/research_agent").glob("*.py")))).hexdigest(),
        "data_sha256": previous["data_sha256"], "corpus_revision": revision,
        "vector_collection": previous["vector_collection"], "baseline_ranking_mismatches": 0,
        "metrics": metrics, "paired_differences": differences,
        "timing": {"reranker_device": args.device, "embedding_device": "cpu",
                   "query_embedding_batch_seconds_including_load": round(embedding_seconds, 3),
                   "rerank_ms": {mode: {"p50": statistics.median(values),
                                        "p95": sorted(values)[int(0.95 * (len(values) - 1))],
                                        "max": max(values)} for mode, values in times.items()},
                   "total_seconds": round(time.perf_counter() - started, 2)},
        "limitations": ["Known-paper dev text evidence subset; no official test, answer or refusal evaluation",
                        "Query vectors precomputed; reranker ran separately for every arm",
                        "MS MARCO trained English model, QASPER domain shift; no fine tuning",
                        "Single process, sequential evaluation; not online concurrency or latency SLA"],
        "cases": cases,
    }
    (ROOT / "reports/p2b_reranker_dev.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
