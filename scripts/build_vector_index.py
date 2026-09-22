"""Build real local embeddings from PostgreSQL text only (no gold files)."""
import argparse
import json
from pathlib import Path
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_agent.embeddings import LocalEncoder
from research_agent.settings import Settings
from research_agent.storage import Repository
from research_agent.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    repo = Repository(Settings.from_env())
    repo.migrate()
    report = VectorStore(repo).build(LocalEncoder(args.device), progress=lambda n, total: print(f"Indexed {n}/{total}", flush=True))
    report["device"] = args.device
    if report["outcome"] == "built":
        (ROOT / "reports/p2b_vector_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
