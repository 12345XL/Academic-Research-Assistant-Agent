CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE vector_collections (
    collection_id text PRIMARY KEY,
    corpus_revision bigint NOT NULL,
    config jsonb NOT NULL,
    state text NOT NULL CHECK (state IN ('building', 'ready')),
    paragraph_count integer NOT NULL DEFAULT 0,
    window_count integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    ready_at timestamptz
);

-- Evidence IDs remain paragraphs; multiple windows represent long paragraphs.
CREATE TABLE paragraph_vectors (
    collection_id text NOT NULL REFERENCES vector_collections(collection_id),
    version_id uuid NOT NULL,
    chunk_id text NOT NULL,
    window_index integer NOT NULL CHECK (window_index >= 0),
    text_sha256 text NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (collection_id, version_id, chunk_id, window_index),
    FOREIGN KEY (version_id, chunk_id) REFERENCES paragraphs(version_id, chunk_id)
);
-- No ANN index: exact search inside the specified paper is the first baseline.
