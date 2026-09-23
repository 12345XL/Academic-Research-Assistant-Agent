-- Metadata only: no raw question, evidence, draft, answer or provider credential.
-- No foreign key to corpus tables: run history survives corpus publications.
CREATE TABLE answer_runs (
    run_id text PRIMARY KEY CHECK (run_id ~ '^[0-9a-f]{32}$'),
    owner_id text NOT NULL,
    paper_id text NOT NULL,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    state text NOT NULL DEFAULT 'running'
        CHECK (state IN ('running','completed','abstained','blocked','failed','interrupted')),
    reason text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    metadata jsonb NOT NULL DEFAULT '{}',
    snapshot jsonb NOT NULL,
    cancel_requested boolean NOT NULL DEFAULT false,
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (jsonb_typeof(snapshot) = 'object'),
    CHECK ((state = 'running' AND reason IS NULL) OR (state <> 'running' AND reason IS NOT NULL))
);
CREATE INDEX answer_runs_owner_created ON answer_runs(owner_id, created_at DESC, run_id DESC);
CREATE INDEX answer_runs_running ON answer_runs(run_id) WHERE state = 'running';
