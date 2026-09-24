-- Uploaded documents share the fact tables, but are not QASPER index members.
ALTER TABLE paper_versions DROP CONSTRAINT paper_versions_split_check;
ALTER TABLE paper_versions ADD CHECK (split IN ('train', 'dev', 'uploaded'));
CREATE TABLE pdf_documents (
    paper_id text PRIMARY KEY REFERENCES papers(paper_id),
    original_object_key text NOT NULL REFERENCES stored_objects(object_key),
    original_sha256 text NOT NULL CHECK (original_sha256 ~ '^[0-9a-f]{64}$'),
    parser_version text NOT NULL,
    page_count integer NOT NULL CHECK (page_count BETWEEN 1 AND 80),
    warnings jsonb NOT NULL,
    UNIQUE(original_sha256, parser_version)
);
