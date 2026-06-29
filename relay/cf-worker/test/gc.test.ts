import { env } from "cloudflare:test";
import { describe, it, expect, beforeEach } from "vitest";
import { Store, ENVELOPE_TTL_MS } from "../src/storage";
import { applySchema } from "./apply-schema";

beforeEach(async () => { await applySchema(env.DB); });

const DAY_MS = 24 * 60 * 60 * 1000;
const NOW = 1_000_000_000_000; // fixed reference epoch

describe("Store.gc", () => {
  it("deletes envelopes older than 7 days, keeps fresh ones", async () => {
    const s = new Store(env.DB);
    // Push a stale envelope (8 days old)
    const staleAt = NOW - 8 * DAY_MS;
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,NULL,0,?)"
    ).bind("mbx-stale", staleAt).run();
    await env.DB.prepare(
      "INSERT INTO envelopes (id, mailbox_id, envelope_json, queued_at) VALUES (?,?,?,?)"
    ).bind("env-stale", "mbx-stale", '{"relay_mailbox":"mbx-stale"}', staleAt).run();

    // Push a fresh envelope (1 day old)
    const freshAt = NOW - 1 * DAY_MS;
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,NULL,0,?)"
    ).bind("mbx-fresh", freshAt).run();
    await env.DB.prepare(
      "INSERT INTO envelopes (id, mailbox_id, envelope_json, queued_at) VALUES (?,?,?,?)"
    ).bind("env-fresh", "mbx-fresh", '{"relay_mailbox":"mbx-fresh"}', freshAt).run();

    const summary = await s.gc(NOW);

    expect(summary.envelopes).toBe(1);

    // stale envelope is gone
    const staleRow = await env.DB.prepare("SELECT 1 FROM envelopes WHERE id=?")
      .bind("env-stale").first();
    expect(staleRow).toBeNull();

    // fresh envelope is kept
    const freshRow = await env.DB.prepare("SELECT 1 FROM envelopes WHERE id=?")
      .bind("env-fresh").first();
    expect(freshRow).not.toBeNull();
  });

  it("deletes expired nonces, keeps live ones", async () => {
    const s = new Store(env.DB);
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,?,1,?)"
    ).bind("mbx-n", "pubA", NOW - DAY_MS).run();

    // Insert an expired nonce (expires_at in the past)
    await env.DB.prepare(
      "INSERT INTO nonces (mailbox_id, nonce, expires_at) VALUES (?,?,?)"
    ).bind("mbx-n", "nonce-expired", NOW - 1).run();

    // Insert a live nonce (expires_at in the future)
    await env.DB.prepare(
      "INSERT INTO nonces (mailbox_id, nonce, expires_at) VALUES (?,?,?)"
    ).bind("mbx-n", "nonce-live", NOW + 60_000).run();

    const summary = await s.gc(NOW);

    expect(summary.nonces).toBe(1);

    const expiredRow = await env.DB.prepare("SELECT 1 FROM nonces WHERE nonce=?")
      .bind("nonce-expired").first();
    expect(expiredRow).toBeNull();

    const liveRow = await env.DB.prepare("SELECT 1 FROM nonces WHERE nonce=?")
      .bind("nonce-live").first();
    expect(liveRow).not.toBeNull();
  });

  it("deletes stale empty unregistered mailboxes, keeps those with envelopes or within TTL", async () => {
    const s = new Store(env.DB);
    const staleAt = NOW - 8 * DAY_MS;
    const freshAt = NOW - 1 * DAY_MS;

    // Stale empty unregistered — should be deleted
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,NULL,0,?)"
    ).bind("mbx-stale-empty", staleAt).run();

    // Fresh empty unregistered — keep
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,NULL,0,?)"
    ).bind("mbx-fresh-empty", freshAt).run();

    // Stale unregistered with a FRESH envelope — keep (the envelope is < 7 days old so it won't be GC'd)
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,NULL,0,?)"
    ).bind("mbx-stale-has-env", staleAt).run();
    await env.DB.prepare(
      "INSERT INTO envelopes (id, mailbox_id, envelope_json, queued_at) VALUES (?,?,?,?)"
    ).bind("env-fresh-has", "mbx-stale-has-env", '{"relay_mailbox":"mbx-stale-has-env"}', freshAt).run();

    // Registered mailbox (stale, empty) — NOT touched by this GC sweep
    await env.DB.prepare(
      "INSERT INTO mailboxes (mailbox_id, identity_pub, registered, created_at) VALUES (?,?,1,?)"
    ).bind("mbx-registered", "pubR", staleAt).run();

    const summary = await s.gc(NOW);

    expect(summary.mailboxes).toBe(1);

    expect(await env.DB.prepare("SELECT 1 FROM mailboxes WHERE mailbox_id=?").bind("mbx-stale-empty").first()).toBeNull();
    expect(await env.DB.prepare("SELECT 1 FROM mailboxes WHERE mailbox_id=?").bind("mbx-fresh-empty").first()).not.toBeNull();
    expect(await env.DB.prepare("SELECT 1 FROM mailboxes WHERE mailbox_id=?").bind("mbx-stale-has-env").first()).not.toBeNull();
    expect(await env.DB.prepare("SELECT 1 FROM mailboxes WHERE mailbox_id=?").bind("mbx-registered").first()).not.toBeNull();
  });

  it("returns zeroes when nothing to clean up", async () => {
    const s = new Store(env.DB);
    const summary = await s.gc(NOW);
    expect(summary).toEqual({ envelopes: 0, nonces: 0, mailboxes: 0 });
  });

  it("ENVELOPE_TTL_MS is 7 days", () => {
    expect(ENVELOPE_TTL_MS).toBe(7 * 24 * 60 * 60 * 1000);
  });
});
