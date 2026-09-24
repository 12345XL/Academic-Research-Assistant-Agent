-- Opt-in working memory, separate from metadata-only execution traces.
CREATE TABLE conversations (
    conversation_id text PRIMARY KEY,
    owner_id text NOT NULL,
    paper_id text NOT NULL,
    paper_version text NOT NULL,
    revision integer NOT NULL DEFAULT 0 CHECK (revision >= 0),
    turns jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(turns) = 'array'),
    expires_at timestamptz NOT NULL
);
CREATE INDEX conversations_expiry ON conversations(expires_at);
CREATE INDEX conversations_owner ON conversations(owner_id);
