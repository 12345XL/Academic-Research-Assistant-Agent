-- Dataset release (qasper-v0.3) is NOT an arXiv paper revision.
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id uuid PRIMARY KEY,
    source text NOT NULL,
    manifest_sha256 text CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    result jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_code text,
    CHECK ((status = 'running') = (finished_at IS NULL))
);

CREATE TABLE IF NOT EXISTS stored_objects (
    object_key text PRIMARY KEY,
    bucket text NOT NULL,
    sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    content_type text NOT NULL,
    state text NOT NULL CHECK (state IN ('staged', 'published', 'orphaned')),
    created_by_job uuid REFERENCES ingestion_jobs(job_id),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id text PRIMARY KEY,
    current_version_id uuid,
    in_current_corpus boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_versions (
    version_id uuid PRIMARY KEY,
    paper_id text NOT NULL REFERENCES papers(paper_id),
    dataset_version text NOT NULL,
    source text NOT NULL,
    split text NOT NULL CHECK (split IN ('train', 'dev')),
    title text NOT NULL,
    abstract text NOT NULL,
    object_key text NOT NULL REFERENCES stored_objects(object_key),
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    state text NOT NULL CHECK (state IN ('active', 'archived')),
    paper_payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (paper_id, version_id),
    UNIQUE (paper_id, content_sha256)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_version_per_paper
    ON paper_versions(paper_id) WHERE state = 'active';
ALTER TABLE papers ADD CONSTRAINT paper_current_version_fk
    FOREIGN KEY (paper_id, current_version_id)
    REFERENCES paper_versions(paper_id, version_id) DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE IF NOT EXISTS paragraphs (
    version_id uuid NOT NULL REFERENCES paper_versions(version_id),
    chunk_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    section_index integer NOT NULL,
    paragraph_index integer NOT NULL CHECK (paragraph_index >= 0),
    section_name text NOT NULL,
    text text NOT NULL,
    text_sha256 text NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    paragraph_payload jsonb NOT NULL,
    PRIMARY KEY (version_id, chunk_id),
    UNIQUE (version_id, ordinal)
);

CREATE TABLE IF NOT EXISTS corpus_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0),
    manifest_sha256 text CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    paper_count integer NOT NULL DEFAULT 0,
    paragraph_count integer NOT NULL DEFAULT 0,
    published_at timestamptz,
    published_by_job uuid REFERENCES ingestion_jobs(job_id)
);
INSERT INTO corpus_state(singleton) VALUES (true) ON CONFLICT DO NOTHING;
