-- Content is saved only on an explicit feedback submission, separate from traces.
CREATE TABLE answer_feedback (
    run_id text PRIMARY KEY REFERENCES answer_runs(run_id),
    revision integer NOT NULL CHECK (revision > 0),
    rating text NOT NULL CHECK (rating IN ('helpful', 'problem')),
    note text NOT NULL DEFAULT '' CHECK (char_length(note) <= 2000),
    target_sha256 text NOT NULL CHECK (target_sha256 ~ '^[0-9a-f]{64}$'),
    target jsonb NOT NULL CHECK (jsonb_typeof(target) = 'object'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
