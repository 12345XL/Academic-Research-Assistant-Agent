"""Offline paired comparison through the production retrieval/fact-check service."""
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
from research_agent.dataset import read_jsonl, sha256_file
from research_agent import __version__
from research_agent.embeddings import CONFIG, LocalEncoder, collection_id
from research_agent.evaluate import gold_references, retrieval_metrics
from research_agent.service import PersistentEvidenceService
from research_agent.settings import Settings
from research_agent.storage import Repository
from research_agent.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    frozen = json.loads((ROOT / "reports/p1_retrieval_dev.json").read_text())
    for filename, expected in frozen["data_sha256"].items():
        if sha256_file(ROOT / "data/processed" / filename) != expected:
            raise ValueError("Evaluation data no longer matches frozen P1")
    questions = {q["question_id"]: q for q in read_jsonl(ROOT / "data/processed/questions.jsonl")}
    selected = [questions[case["question_id"]] for case in frozen["cases"]]
    assert len(selected) == 745 and all(q["split"] == "dev" for q in selected)
    repo = Repository(Settings.from_env())
    revision = repo.revision()
    assert VectorStore(repo).status(revision)["state"] == "ready"
    encoder = LocalEncoder(args.device)
    started = time.perf_counter()
    query_vectors = encoder.encode_queries([q["question"] for q in selected])
    embedding_seconds = time.perf_counter() - started
    service = PersistentEvidenceService(repo, encoder)
    cases = []
    latencies = {mode: [] for mode in ("bm25", "dense", "hybrid")}
    for number, (question, baseline, vector) in enumerate(zip(selected, frozen["cases"], query_vectors), 1):
        references, reason = gold_references(question)
        assert reason is None
        case = {"question_id": question["question_id"], "paper_id": question["paper_id"]}
        for mode in latencies:
            result = service.retrieve(question["question"], question["paper_id"], 5, mode=mode, query_vector=vector)
            assert result["trace"]["corpus_revision"] == str(revision)
            ranked = [item["chunk_id"] for item in result["citations"]]
            if mode == "bm25" and ranked != baseline["retrieved_chunk_ids"]:
                raise ValueError("BM25 drift from frozen P1")
            case[mode] = {"retrieved_chunk_ids": ranked, **retrieval_metrics(ranked, references)}
            latencies[mode].append(result["trace"]["latency_ms"])
        cases.append(case)
        if number % 50 == 0:
            print(f"Evaluated {number}/{len(selected)}", flush=True)
    assert repo.revision() == revision
    metrics = {mode: {f"{metric}_at_5": statistics.mean(case[mode][metric] for case in cases)
                      for metric in ("hit", "recall", "mrr")} for mode in latencies}
    differences = {}
    for mode in ("dense", "hybrid"):
        differences[mode] = {
            "new_hits": sum(c[mode]["hit"] > c["bm25"]["hit"] for c in cases),
            "lost_hits": sum(c[mode]["hit"] < c["bm25"]["hit"] for c in cases),
            "ranking_changed": sum(c[mode]["retrieved_chunk_ids"] != c["bm25"]["retrieved_chunk_ids"] for c in cases),
        }
    report = {
        "stage": "P2B-1", "protocol": "specified-paper-text-evidence-paired-retrieval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "questions": len(cases),
        "app_version": __version__,
        "source_sha256": hashlib.sha256(b"".join(
            path.name.encode() + path.read_bytes() for path in sorted((ROOT / "src/research_agent").glob("*.py"))
        )).hexdigest(),
        "data_sha256": frozen["data_sha256"], "corpus_revision": revision,
        "vector_collection": collection_id(revision), "config": CONFIG,
        "retrieval_config": {"top_k": 5, "candidate_pool_per_branch": 20, "rrf_constant": 60},
        "bm25_ranking_mismatches": 0, "metrics": metrics, "differences_from_bm25": differences,
        "timing": {"device": args.device, "query_embedding_batch_seconds_including_load": round(embedding_seconds, 3),
                   "retrieval_p50_ms_excluding_query_encoding": {mode: statistics.median(values) for mode, values in latencies.items()},
                   "total_elapsed_seconds": round(time.perf_counter() - started, 2)},
        "limitations": ["Frozen dev text-reference subset; official test unused; no answer/refusal score",
                        "Query vectors precomputed in a batch; timing is not online end-to-end latency",
                        "English known-paper search, no corpus-wide or Chinese evaluation",
                        "No reranker; RRF parameters not retuned on these results"],
        "cases": cases,
    }
    (ROOT / "reports/p2b_retrieval_dev.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
