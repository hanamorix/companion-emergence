// apply-schema.ts — runs DDL against the ephemeral test D1 binding.
// Uses prepare().run() per statement (not exec()) because miniflare's exec()
// splits naively on ";" and chokes on multi-line CREATE TABLE bodies.
export async function applySchema(db: D1Database): Promise<void> {
  const drops = [
    "DROP TABLE IF EXISTS nonces",
    "DROP TABLE IF EXISTS envelopes",
    "DROP INDEX IF EXISTS idx_env_mbx",
    "DROP TABLE IF EXISTS mailboxes",
  ];
  const creates = [
    `CREATE TABLE IF NOT EXISTS mailboxes (
      mailbox_id TEXT PRIMARY KEY,
      identity_pub TEXT,
      registered INTEGER NOT NULL,
      created_at INTEGER NOT NULL
    )`,
    `CREATE TABLE IF NOT EXISTS envelopes (
      id TEXT PRIMARY KEY,
      mailbox_id TEXT NOT NULL,
      envelope_json TEXT NOT NULL,
      queued_at INTEGER NOT NULL
    )`,
    "CREATE INDEX IF NOT EXISTS idx_env_mbx ON envelopes(mailbox_id, queued_at)",
    `CREATE TABLE IF NOT EXISTS nonces (
      mailbox_id TEXT NOT NULL,
      nonce TEXT NOT NULL,
      expires_at INTEGER NOT NULL,
      PRIMARY KEY (mailbox_id, nonce)
    )`,
  ];
  for (const sql of [...drops, ...creates]) {
    await db.prepare(sql).run();
  }
}
