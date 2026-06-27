CREATE TABLE IF NOT EXISTS mailboxes (
  mailbox_id TEXT PRIMARY KEY,
  identity_pub TEXT,                 -- NULL = unregistered (pushed-to only)
  registered INTEGER NOT NULL,       -- 0 unregistered, 1 registered
  created_at INTEGER NOT NULL        -- ms epoch (insertion order for eviction)
);
CREATE TABLE IF NOT EXISTS envelopes (
  id TEXT PRIMARY KEY,
  mailbox_id TEXT NOT NULL,
  envelope_json TEXT NOT NULL,
  queued_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_env_mbx ON envelopes(mailbox_id, queued_at);
CREATE TABLE IF NOT EXISTS nonces (
  mailbox_id TEXT NOT NULL,
  nonce TEXT NOT NULL,
  expires_at INTEGER NOT NULL,       -- ms epoch
  PRIMARY KEY (mailbox_id, nonce)
);
