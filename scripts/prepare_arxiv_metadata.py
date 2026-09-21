"""Freeze arXiv submission dates and optional journal references for QASPER IDs.

This is metadata for library browsing, never retrieval content or evaluation gold.
arXiv's <published> field is first submission time, not journal publication time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ID = re.compile(r"^\d{2}(?:0[1-9]|1[0-2])\.\d{5}$")
ENTRY_ID = re.compile(r"/abs/(\d{4}\.\d{5})(?:v\d+)?$")
API = "https://export.arxiv.org/api/query"


def get_feed(ids: list[str]) -> dict[str, dict]:
    query = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    request = urllib.request.Request(f"{API}?{query}", headers={
        "User-Agent": "AcademicResearchAssistant/0.3 (noncommercial educational project)",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(4_000_001)
    if len(content) > 4_000_000:
        raise ValueError("arXiv metadata response exceeded the per-batch size limit")
    root = ET.fromstring(content)
    output = {}
    for entry in root.findall(f"{ATOM}entry"):
        match = ENTRY_ID.search(entry.findtext(f"{ATOM}id", default=""))
        if not match or match.group(1) not in ids:
            continue
        paper_id = match.group(1)
        published = entry.findtext(f"{ATOM}published")
        if not published:
            raise ValueError(f"Missing arXiv first-submission date: {paper_id}")
        datetime.fromisoformat(published.replace("Z", "+00:00"))
        output[paper_id] = {
            "paper_id": paper_id,
            "arxiv_submitted_at": published,
            "journal_ref": (entry.findtext(f"{ARXIV}journal_ref") or "").strip(),
            "doi": (entry.findtext(f"{ARXIV}doi") or "").strip(),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "metadata/qasper_arxiv.json")
    args = parser.parse_args()
    papers = [json.loads(line) for line in (ROOT / "data/processed/papers.jsonl").open(encoding="utf-8")]
    ids = sorted(p["paper_id"] for p in papers)
    if len(set(ids)) != len(ids) or any(not ID.fullmatch(paper_id) for paper_id in ids):
        parser.error("QASPER paper IDs must be unique modern arXiv identifiers")
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if set(existing["papers"]) == set(ids):
            print(f"Existing arXiv metadata snapshot covers {len(ids)} papers; use a new output path to refresh")
            return
    found = {}
    for start in range(0, len(ids), 200):
        if start:
            time.sleep(3)  # arXiv API asks callers to pause between requests.
        batch = ids[start:start + 200]
        for attempt in range(3):
            try:
                found.update(get_feed(batch))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        print(f"arXiv metadata: {min(start + 200, len(ids))}/{len(ids)} requested, {len(found)} found", flush=True)
    missing = sorted(set(ids) - set(found))
    result = {
        "source": API,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "meaning": "arxiv_submitted_at is first arXiv submission, not formal publication",
        "papers": {paper_id: found.get(paper_id) for paper_id in ids},
        "missing_ids": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {len(found)} metadata records; {len(missing)} missing")


if __name__ == "__main__":
    main()
