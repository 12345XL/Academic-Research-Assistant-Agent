"""Freeze stratified offline train/dev manifests without invoking models."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_agent.dataset import read_jsonl, sha256_file
from research_agent.evaluation_design import build_manifest, freeze_manifest


def main():
    data = ROOT / "data/processed"
    known = json.loads((ROOT / "reports/p2b_generation_selection.json").read_text())
    manifest = build_manifest(
        read_jsonl(data / "questions.jsonl"),
        {name: sha256_file(data / name) for name in ("questions.jsonl", "paragraphs.jsonl", "papers.jsonl")},
        {case["question_id"] for case in known["cases"]},
    )
    result = freeze_manifest(ROOT / "reports/p2b_generation_strata_v2.json", manifest)
    print(json.dumps({"freeze": result, "status": manifest["status"], "model_calls": 0,
                      "inventory": manifest["inventory"],
                      "selected": {name: pool["selected_by_category"] for name, pool in manifest["pools"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
