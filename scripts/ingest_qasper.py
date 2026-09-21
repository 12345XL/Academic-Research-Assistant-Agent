"""Persist the prepared public corpus; never reads questions.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from research_agent.ingestion import CorpusValidationError, ingest_qasper
from research_agent.settings import ConfigurationError, Settings
from research_agent.storage import Repository, S3ObjectStore, StorageError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--migrate", action="store_true", help="Apply pending SQL migrations first")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    try:
        settings = Settings.from_env()
        repo, store = Repository(settings), S3ObjectStore(settings)
        if args.migrate:
            repo.migrate()
        result = ingest_qasper(repo, store, args.data_dir)
    except (ConfigurationError, CorpusValidationError, StorageError, FileNotFoundError) as exc:
        parser.exit(1, f"Import error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
