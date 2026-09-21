"""Load frozen arXiv metadata and conservative CCF venue labels into PostgreSQL."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.paper_metadata import (
    CATALOG_CHECKED_AT,
    ccf_venue_from_journal_ref,
    research_direction_from_primary_category,
)
from research_agent.settings import Settings
from research_agent.storage import Repository


def build_rows(snapshot: dict, paper_ids: set[str]) -> list[dict]:
    if snapshot.get("source") != "https://export.arxiv.org/api/query":
        raise ValueError("Unexpected metadata source")
    entries = snapshot.get("papers")
    if not isinstance(entries, dict) or set(entries) != paper_ids or snapshot.get("missing_ids"):
        raise ValueError("Metadata snapshot does not cover the current paper corpus")
    rows = []
    for paper_id in sorted(paper_ids):
        item = entries[paper_id]
        if not isinstance(item, dict) or item.get("paper_id") != paper_id:
            raise ValueError("Invalid metadata paper identity")
        published = item.get("arxiv_submitted_at")
        if not isinstance(published, str):
            raise ValueError("Missing arXiv first-submission date")
        date = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("arXiv first-submission date must include a timezone")
        journal_ref, doi = item.get("journal_ref"), item.get("doi")
        if not isinstance(journal_ref, str) or not isinstance(doi, str):
            raise ValueError("Malformed arXiv journal reference or DOI")
        primary_category = item.get("primary_category")
        categories = item.get("categories")
        pdf_url = item.get("pdf_url")
        if (not isinstance(primary_category, str) or not primary_category
                or not isinstance(categories, list) or primary_category not in categories
                or any(not isinstance(category, str) or not category for category in categories)):
            raise ValueError("Malformed arXiv subject categories")
        if not isinstance(pdf_url, str) or not pdf_url.startswith("https://arxiv.org/pdf/"):
            raise ValueError("Malformed arXiv PDF URL")
        venue = ccf_venue_from_journal_ref(journal_ref) or {}
        if venue:
            venue = {"ccf_venue": venue["venue"], "ccf_level": venue["ccf_level"],
                     "ccf_catalog_url": venue["ccf_catalog_url"]}
        direction = research_direction_from_primary_category(primary_category)
        rows.append({
            "paper_id": paper_id, "arxiv_submitted_at": date, "journal_ref": journal_ref,
            "doi": doi, "primary_category": primary_category, "categories": categories,
            "pdf_url": pdf_url, **direction, **venue, "metadata_source": snapshot["source"],
            "metadata_checked_at": datetime.fromisoformat(snapshot["retrieved_at"]),
        })
    return rows


def main() -> None:
    load_dotenv(ROOT / ".env")
    repo = Repository(Settings.from_env())
    repo.migrate()
    with repo.connect() as conn:
        paper_ids = {r["paper_id"] for r in conn.execute("SELECT paper_id FROM papers WHERE in_current_corpus")}
    snapshot_path = ROOT / "metadata/qasper_arxiv.json"
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    rows = build_rows(snapshot, paper_ids)
    repo.upsert_paper_metadata(rows)
    summary = repo.summary()
    levels = Counter(r.get("ccf_level") or "unclassified" for r in rows)
    directions = Counter(r["research_direction"] for r in rows)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "paper_count": len(rows),
        "arxiv_submitted_dates": summary["metadata_dates"], "ccf_venues": summary["metadata_ccf"],
        "arxiv_primary_categories": summary["metadata_categorized"], "arxiv_pdf_links": len(rows),
        "ccf_level_counts": {level: levels[level] for level in ("A", "B", "C", "unclassified")},
        "research_direction_counts": dict(sorted(directions.items())),
        "date_source": snapshot["source"], "date_snapshot_at": snapshot["retrieved_at"],
        "date_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "ccf_catalog_checked_at": CATALOG_CHECKED_AT,
        "direction_qualification": "Broad browsing directions are deterministic groups of official arXiv "
                                   "primary categories; cross-lists do not create duplicate memberships.",
        "pdf_qualification": "PDF URLs point to arXiv and are not copied to or checksum-verified by local storage.",
        "qualification": "CCF labels are venue tiers inferred conservatively from arXiv journal references; "
                         "they do not certify an individual paper's full/regular status or quality.",
    }
    target = ROOT / "reports/paper_sort_metadata.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
