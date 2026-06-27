import { env } from "cloudflare:test";
import { describe, it, expect, beforeEach } from "vitest";
import { Store, RelayReject } from "../src/storage";
import { applySchema } from "./apply-schema";

beforeEach(async () => { await applySchema(env.DB); });

describe("Store", () => {
  it("register is first-write-wins and promotes an unregistered key", async () => {
    const s = new Store(env.DB);
    await s.push({ relay_mailbox: "m1", x: 1 }, 1000);      // m1 now unregistered
    expect(await s.isRegistered("m1")).toBe(false);
    expect(await s.register("m1", "pubA", 2000)).toBe("pubA");
    expect(await s.isRegistered("m1")).toBe(true);
    expect(await s.register("m1", "pubB", 3000)).toBe("pubA"); // first-write-wins
  });

  it("push/fetch/ack round-trips and keeps id separate from envelope", async () => {
    const s = new Store(env.DB);
    const id = await s.push({ relay_mailbox: "m2", body: "opaque" }, 1000);
    const got = await s.fetch("m2");
    expect(got).toEqual([{ id, envelope: { relay_mailbox: "m2", body: "opaque" } }]);
    await s.ack("m2", [id]);
    expect(await s.fetch("m2")).toEqual([]);
  });

  it("nonce lifecycle: issue → live → discard; TTL expiry", async () => {
    const s = new Store(env.DB);
    await s.register("m3", "pubA", 1000);
    await s.issueNonce("m3", "n1", 1000);
    expect(await s.nonceLive("m3", "n1", 1000)).toBe(true);
    expect(await s.nonceLive("m3", "n1", 1000 + 121_000)).toBe(false); // 120s TTL
    await s.issueNonce("m3", "n2", 2000);
    await s.discardNonce("m3", "n2");
    expect(await s.nonceLive("m3", "n2", 2000)).toBe(false);
  });

  it("caps: envelope too large → 413; queue depth → 429", async () => {
    const s = new Store(env.DB);
    const big = { relay_mailbox: "m4", body: "z".repeat(70_000) };
    await expect(s.push(big, 1000)).rejects.toMatchObject({ status: 413 } as RelayReject);
    for (let i = 0; i < 256; i++) await s.push({ relay_mailbox: "m5", n: i }, 1000 + i);
    await expect(s.push({ relay_mailbox: "m5", n: 256 }, 2000)).rejects.toMatchObject({ status: 429 });
  });

  it("register atomic first-write-wins: second register with different key never overwrites", async () => {
    const s = new Store(env.DB);
    // First registration — must win and persist pubA.
    const r1 = await s.register("mbx-race", "pubA", 1000);
    expect(r1).toBe("pubA");
    expect(await s.ownerOf("mbx-race")).toBe("pubA");

    // Second registration with a different key — must return the existing winner (pubA)
    // and must NOT overwrite the stored identity_pub.
    const r2 = await s.register("mbx-race", "pubB", 2000);
    expect(r2).toBe("pubA");
    expect(await s.ownerOf("mbx-race")).toBe("pubA"); // no overwrite
  });

  it("evictOldestEmptyUnregistered: 256 empty unregistered + push to 257th evicts oldest empty", async () => {
    const s = new Store(env.DB);
    const IDS: string[] = [];
    // Fill the unregistered pool to MAX_UNREGISTERED (256): each mailbox gets
    // one envelope, then that envelope is acked so the queue is empty.
    // created_at increases with i so the oldest-created is IDS[0].
    for (let i = 0; i < 256; i++) {
      const mbx = `evict-mbx-${String(i).padStart(4, "0")}`;
      IDS.push(mbx);
      const envId = await s.push({ relay_mailbox: mbx, n: i }, 1000 + i);
      await s.ack(mbx, [envId]); // queue now empty — eligible for eviction
    }

    // Give IDS[0] and IDS[1] a live envelope so they must NOT be evicted.
    await s.push({ relay_mailbox: IDS[0], body: "held" }, 2000);
    await s.push({ relay_mailbox: IDS[1], body: "also-held" }, 2001);

    // Push to a 257th NEW mailbox. Pool is full (256 unregistered rows).
    // evictOldestEmptyUnregistered must pick the oldest EMPTY mailbox (IDS[2],
    // because IDS[0] and IDS[1] both have live envelopes).
    const newEnvId = await s.push({ relay_mailbox: "evict-mbx-new", x: 1 }, 3000);
    expect(typeof newEnvId).toBe("string"); // succeeded — a slot was freed

    // The new mailbox's envelope is present.
    expect((await s.fetch("evict-mbx-new")).length).toBe(1);

    // Mailboxes with live envelopes were NOT evicted.
    expect((await s.fetch(IDS[0])).length).toBe(1);
    expect((await s.fetch(IDS[1])).length).toBe(1);

    // IDS[2] (the oldest empty unregistered) was evicted — its fetch returns empty.
    expect((await s.fetch(IDS[2])).length).toBe(0);
  });
});
