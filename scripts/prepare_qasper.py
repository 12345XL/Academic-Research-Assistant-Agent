#!/usr/bin/env python3
"""Download fixed QASPER train/dev and create separated corpus/QA artifacts."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.dataset import prepare_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Output project root (default: this repository)")
    args = parser.parse_args()
    audit = prepare_dataset(args.root.resolve())
    print(json.dumps({"version": audit["version"], "splits": audit["splits"],
                      "integrity": audit["integrity"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
