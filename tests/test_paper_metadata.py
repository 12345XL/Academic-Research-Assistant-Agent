from datetime import datetime, timezone

import pytest

from research_agent.paper_metadata import ccf_venue_from_journal_ref, research_direction_from_primary_category
from scripts.ingest_paper_metadata import build_rows


@pytest.mark.parametrize(("reference", "venue", "level"), [
    ("Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics", "ACL", "A"),
    ("EMNLP 2019 1586-1595", "EMNLP", "B"),
    ("Computational Linguistics 43(1), 2017", "Computational Linguistics", "B"),
    ("Journal of Machine Learning Research 21 (2020)", "JMLR", "A"),
    ("CoNLL'2019", "CoNLL", "C"),
])
def test_ccf_catalog_matches_unambiguous_venues(reference, venue, level):
    result = ccf_venue_from_journal_ref(reference)
    assert result["venue"] == venue
    assert result["ccf_level"] == level
    assert result["ccf_catalog_url"].startswith("https://www.ccf.org.cn/")


@pytest.mark.parametrize("reference", [
    "",
    "3rd Conversational AI Workshop at NeurIPS 2019",
    "NLP4IF@EMNLP-2019",
    "NAACL 2019, Volume 1 (Long and Short Papers)",
    "Unknown Journal 12 (2020)",
])
def test_ccf_catalog_does_not_promote_unverified_or_satellite_venues(reference):
    assert ccf_venue_from_journal_ref(reference) is None


def test_metadata_snapshot_must_exactly_cover_current_corpus():
    item = {
        "paper_id": "1503.00841", "arxiv_submitted_at": "2015-03-03T06:59:28Z",
        "journal_ref": "", "doi": "", "primary_category": "cs.CL",
        "categories": ["cs.CL", "cs.LG"], "pdf_url": "https://arxiv.org/pdf/1503.00841v1",
    }
    snapshot = {
        "source": "https://export.arxiv.org/api/query",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "papers": {"1503.00841": item}, "missing_ids": [],
    }
    rows = build_rows(snapshot, {"1503.00841"})
    assert rows[0]["arxiv_submitted_at"].year == 2015
    assert rows[0]["research_direction"] == "nlp"
    assert rows[0].get("ccf_level") is None
    with pytest.raises(ValueError, match="does not cover"):
        build_rows(snapshot, {"1503.00841", "1601.00901"})


@pytest.mark.parametrize(("category", "direction"), [
    ("cs.CL", "nlp"), ("cs.LG", "machine_learning"), ("cs.IR", "information_retrieval"),
    ("eess.AS", "speech_audio"), ("cs.CV", "computer_vision_multimedia"),
    ("q-fin.ST", "other"),
])
def test_research_direction_uses_official_primary_category(category, direction):
    result = research_direction_from_primary_category(category)
    assert result["research_direction"] == direction
    assert result["research_direction_label"]
