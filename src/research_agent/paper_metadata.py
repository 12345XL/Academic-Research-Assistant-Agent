"""Conservative venue-level CCF labels from arXiv journal references.

The CCF catalog ranks venues, not individual manuscripts. A journal reference
does not prove a conference paper was a full/regular paper. We therefore expose
these labels as venue metadata with provenance, never as paper quality scores.
"""
from __future__ import annotations

import re

CCF_AI_URL = "https://www.ccf.org.cn/Academic_Evaluation/AI/"
CCF_DM_URL = "https://www.ccf.org.cn/Academic_Evaluation/DM_CS/"
CCF_CROSS_URL = "https://www.ccf.org.cn/Academic_Evaluation/Cross_Compre_Emerging/"
CATALOG_CHECKED_AT = "2026-09-21"

# A reference containing these words may refer to a satellite or non-regular
# contribution. Do not transfer the parent conference's tier to it.
EXCLUDED_CONFERENCE = re.compile(
    r"\b(workshop|findings|short papers?|demo(?:nstration)?s?|poster|"
    r"tutorial|companion|doctoral consortium|extended abstract|google sites)\b|"
    r"\b\w+@(?:EMNLP|ACL|NAACL|NeurIPS|NIPS|ICCV|CVPR)\b", re.I
)

JOURNALS = (
    (r"^Journal of Machine Learning Research\b", "JMLR", "A", CCF_AI_URL),
    (r"^IEEE Transactions on Pattern Analysis and Machine Intelligence\b", "TPAMI", "A", CCF_AI_URL),
    (r"^Artificial Intelligence\s*(?:[,;(]|\d)", "AI", "A", CCF_AI_URL),
    (r"^Transactions of the Association for Computational Linguistics\b", "TACL", "B", CCF_AI_URL),
    (r"^Computational Linguistics\s*(?:[,;(]|\d)", "Computational Linguistics", "B", CCF_AI_URL),
    (r"^Journal of Artificial Intelligence Research\b", "JAIR", "B", CCF_AI_URL),
    (r"^IEEE/ACM Transactions on Audio, Speech, and Language Processing\b", "TASLP", "B", CCF_AI_URL),
    (r"^Neural Networks\s*(?:[,;(]|\d)", "Neural Networks", "B", CCF_AI_URL),
)

CONFERENCES = (
    (r"\bNAACL\b|North American Chapter of the Association for Computational Linguistics", "NAACL", "B", CCF_AI_URL),
    (r"\bEMNLP\b|Conference on Empirical Methods in Natural Language Processing", "EMNLP", "B", CCF_AI_URL),
    (r"\bCOLING\b|International Conference on Computational Linguistics(?! and)", "COLING", "B", CCF_AI_URL),
    (r"\bACL\b|Annual Meeting of the Association for Computational Linguistics", "ACL", "A", CCF_AI_URL),
    (r"\bAAAI(?:[- ]\d{2,4})?\b|AAAI Conference on Artificial Intelligence", "AAAI", "A", CCF_AI_URL),
    (r"\bNeurIPS\b|\bNIPS\b|Conference on Neural Information Processing Systems", "NeurIPS", "A", CCF_AI_URL),
    (r"\bICML\b|International Conference on Machine Learning", "ICML", "A", CCF_AI_URL),
    (r"\bIJCAI\b|International Joint Conference on Artificial Intelligence", "IJCAI", "A", CCF_AI_URL),
    (r"\bECAI\b|European Conference on Artificial Intelligence", "ECAI", "B", CCF_AI_URL),
    (r"\bCoNLL\b|Conference on Computational Natural Language Learning", "CoNLL", "C", CCF_AI_URL),
    (r"\bAISTATS\b|International Conference on Artificial Intelligence and Statistics", "AISTATS", "C", CCF_AI_URL),
    (r"\bSIGIR\b|International ACM SIGIR Conference", "SIGIR", "A", CCF_DM_URL),
    (r"\bCIKM\b|Conference on Information and Knowledge Management", "CIKM", "B", CCF_DM_URL),
    (r"\bWSDM\b|Conference on Web Search and Data Mining", "WSDM", "B", CCF_DM_URL),
    (r"\bISWC\b|International Semantic Web Conference", "ISWC", "B", CCF_DM_URL),
    (r"\bWWW\b|The Web Conference", "WWW", "A", CCF_CROSS_URL),
)


def ccf_venue_from_journal_ref(reference: str) -> dict[str, str] | None:
    """Return a catalog-matched venue only when the reference is unambiguous."""
    if not isinstance(reference, str) or not reference.strip():
        return None
    for pattern, venue, level, url in JOURNALS:
        if re.search(pattern, reference, re.I):
            return {"venue": venue, "ccf_level": level, "ccf_catalog_url": url}
    if EXCLUDED_CONFERENCE.search(reference) or "Long and Short Papers" in reference:
        return None
    matches = [
        {"venue": venue, "ccf_level": level, "ccf_catalog_url": url}
        for pattern, venue, level, url in CONFERENCES if re.search(pattern, reference, re.I)
    ]
    # Co-located/multi-venue references can be assigned only when all matches
    # share one catalog tier; the displayed venue remains the first match.
    return matches[0] if matches and len({item["ccf_level"] for item in matches}) == 1 else None
