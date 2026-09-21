-- Browsing metadata is separate from the immutable QASPER retrieval corpus.
-- arxiv_submitted_at is first arXiv submission, never formal journal date.
CREATE TABLE IF NOT EXISTS paper_metadata (
    paper_id text PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    arxiv_submitted_at timestamptz,
    arxiv_journal_ref text NOT NULL DEFAULT '',
    arxiv_doi text NOT NULL DEFAULT '',
    ccf_venue text,
    ccf_level text CHECK (ccf_level IN ('A', 'B', 'C')),
    ccf_catalog_url text,
    metadata_source text NOT NULL,
    metadata_checked_at timestamptz NOT NULL,
    CHECK ((ccf_venue IS NULL) = (ccf_level IS NULL)),
    CHECK ((ccf_level IS NULL) = (ccf_catalog_url IS NULL))
);
CREATE INDEX IF NOT EXISTS paper_metadata_submitted_idx ON paper_metadata(arxiv_submitted_at DESC);
CREATE INDEX IF NOT EXISTS paper_metadata_ccf_idx ON paper_metadata(ccf_level, arxiv_submitted_at DESC);
