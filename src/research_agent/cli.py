from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import EvidenceService


def main() -> None:
    parser = argparse.ArgumentParser(description="P1: QASPER 原文证据检索，不调用生成模型")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("papers")
    listing.add_argument("--limit", type=int, default=10)
    search = sub.add_parser("search")
    search.add_argument("--paper-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        service = EvidenceService(args.data_dir)
        if args.command == "papers":
            result = sorted(service.papers.values(), key=lambda p: p["paper_id"])[:args.limit]
        else:
            result = service.retrieve(args.query, args.paper_id, args.top_k)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(2, f"无法执行：{exc}\n先运行 python scripts/prepare_qasper.py 准备数据。\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
