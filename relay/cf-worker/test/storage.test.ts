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
});
