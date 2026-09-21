-- Official arXiv subject metadata supports browsing only; it does not enter RAG chunks.
ALTER TABLE paper_metadata ADD COLUMN IF NOT EXISTS arxiv_primary_category text;
ALTER TABLE paper_metadata ADD COLUMN IF NOT EXISTS arxiv_categories text[] NOT NULL DEFAULT '{}';
ALTER TABLE paper_metadata ADD COLUMN IF NOT EXISTS arxiv_pdf_url text;
ALTER TABLE paper_metadata ADD COLUMN IF NOT EXISTS research_direction text;

ALTER TABLE paper_metadata DROP CONSTRAINT IF EXISTS paper_metadata_pdf_url_check;
ALTER TABLE paper_metadata ADD CONSTRAINT paper_metadata_pdf_url_check
    CHECK (arxiv_pdf_url IS NULL OR arxiv_pdf_url LIKE 'https://arxiv.org/pdf/%');

ALTER TABLE paper_metadata DROP CONSTRAINT IF EXISTS paper_metadata_research_direction_check;
ALTER TABLE paper_metadata ADD CONSTRAINT paper_metadata_research_direction_check CHECK (
    research_direction IS NULL OR research_direction IN (
        'nlp', 'machine_learning', 'information_retrieval', 'artificial_intelligence',
        'speech_audio', 'computer_vision_multimedia', 'social_computing',
        'human_computer_interaction', 'robotics', 'other'
    )
);

CREATE INDEX IF NOT EXISTS paper_metadata_direction_idx
    ON paper_metadata(research_direction, arxiv_submitted_at DESC);
