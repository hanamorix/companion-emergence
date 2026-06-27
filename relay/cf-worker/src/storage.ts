import { canonicalJson } from "./canonical";

export const NONCE_TTL_MS = 120_000;
export const MAX_ENVELOPE_BYTES = 65_536;
export const MAX_QUEUE_DEPTH = 256;
export const MAX_REGISTERED = 1024;
export const MAX_UNREGISTERED = 256;

export class RelayReject extends Error {
  constructor(public status: number, public detail: string) { super(detail); }
}

export class Store {
  constructor(private db: D1Database) {}

  async ownerOf(mailboxId: string): Promise<string | null> {
    const r = await this.db
      .prepare("SELECT identity_pub FROM mailboxes WHERE mailbox_id=? AND registered=1")
      .bind(mailboxId)
      .first<{ identity_pub: string }>();
    return r ? r.identity_pub : null;
  }

  async isRegistered(mailboxId: string): Promise<boolean> {
    return (await this.ownerOf(mailboxId)) !== null;
  }

  async register(mailboxId: string, identityPub: string, nowMs: number): Promise<string> {
    const existing = await this.ownerOf(mailboxId);
    if (existing) return existing;

    const count = await this.db
      .prepare("SELECT COUNT(*) AS c FROM mailboxes WHERE registered=1")
      .first<{ c: number }>();
    if ((count?.c ?? 0) >= MAX_REGISTERED) {
      throw new RelayReject(429, "registered mailbox cap reached");
    }

    await this.db
      .prepare(
        "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,?,1,?) " +
        "ON CONFLICT(mailbox_id) DO UPDATE SET identity_pub=excluded.identity_pub, registered=1"
      )
      .bind(mailboxId, identityPub, nowMs)
      .run();
    return identityPub;
  }

  async issueNonce(mailboxId: string, nonce: string, nowMs: number): Promise<void> {
    await this.db.prepare("DELETE FROM nonces WHERE expires_at<=?").bind(nowMs).run();
    await this.db
      .prepare("INSERT OR REPLACE INTO nonces (mailbox_id,nonce,expires_at) VALUES (?,?,?)")
      .bind(mailboxId, nonce, nowMs + NONCE_TTL_MS)
      .run();
  }

  async nonceLive(mailboxId: string, nonce: string, nowMs: number): Promise<boolean> {
    const r = await this.db
      .prepare("SELECT 1 FROM nonces WHERE mailbox_id=? AND nonce=? AND expires_at>?")
      .bind(mailboxId, nonce, nowMs)
      .first();
    return r !== null;
  }

  async discardNonce(mailboxId: string, nonce: string): Promise<void> {
    await this.db
      .prepare("DELETE FROM nonces WHERE mailbox_id=? AND nonce=?")
      .bind(mailboxId, nonce)
      .run();
  }

  async push(envelope: Record<string, unknown>, nowMs: number): Promise<string> {
    const json = canonicalJson(envelope);
    if (new TextEncoder().encode(json).length > MAX_ENVELOPE_BYTES) {
      throw new RelayReject(413, "envelope too large");
    }

    const mbx = String(envelope["relay_mailbox"]);

    const known = await this.db
      .prepare("SELECT 1 FROM mailboxes WHERE mailbox_id=?")
      .bind(mbx)
      .first();

    if (!known) {
      const u = await this.db
        .prepare("SELECT COUNT(*) AS c FROM mailboxes WHERE registered=0")
        .first<{ c: number }>();
      if ((u?.c ?? 0) >= MAX_UNREGISTERED) {
        if (!(await this.evictOldestEmptyUnregistered())) {
          throw new RelayReject(429, "unregistered mailbox pool full");
        }
      }
      await this.db
        .prepare("INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,NULL,0,?)")
        .bind(mbx, nowMs)
        .run();
    }

    const depth = await this.db
      .prepare("SELECT COUNT(*) AS c FROM envelopes WHERE mailbox_id=?")
      .bind(mbx)
      .first<{ c: number }>();
    if ((depth?.c ?? 0) >= MAX_QUEUE_DEPTH) {
      throw new RelayReject(429, "mailbox queue full");
    }

    const id = "env_" + crypto.randomUUID();
    await this.db
      .prepare("INSERT INTO envelopes (id, mailbox_id, envelope_json, queued_at) VALUES (?,?,?,?)")
      .bind(id, mbx, json, nowMs)
      .run();
    return id;
  }

  private async evictOldestEmptyUnregistered(): Promise<boolean> {
    const row = await this.db
      .prepare(
        "SELECT m.mailbox_id AS id FROM mailboxes m " +
        "WHERE m.registered=0 " +
        "AND NOT EXISTS (SELECT 1 FROM envelopes e WHERE e.mailbox_id=m.mailbox_id) " +
        "ORDER BY m.created_at ASC LIMIT 1"
      )
      .first<{ id: string }>();
    if (!row) return false;
    await this.db.prepare("DELETE FROM mailboxes WHERE mailbox_id=?").bind(row.id).run();
    await this.db.prepare("DELETE FROM nonces WHERE mailbox_id=?").bind(row.id).run();
    return true;
  }

  async fetch(mailboxId: string): Promise<{ id: string; envelope: unknown }[]> {
    const { results } = await this.db
      .prepare("SELECT id, envelope_json FROM envelopes WHERE mailbox_id=? ORDER BY queued_at ASC")
      .bind(mailboxId)
      .all<{ id: string; envelope_json: string }>();
    return (results ?? []).map((r) => ({ id: r.id, envelope: JSON.parse(r.envelope_json) }));
  }

  async ack(mailboxId: string, ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    const ph = ids.map(() => "?").join(",");
    await this.db
      .prepare(`DELETE FROM envelopes WHERE mailbox_id=? AND id IN (${ph})`)
      .bind(mailboxId, ...ids)
      .run();
  }
}
