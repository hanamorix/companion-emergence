import { env, SELF } from "cloudflare:test";
import { describe, it, expect, beforeEach } from "vitest";
import { applySchema } from "./apply-schema";

beforeEach(async () => { await applySchema(env.DB); });
const post = (p: string, b: unknown) => SELF.fetch("https://relay.test" + p, { method: "POST", body: JSON.stringify(b), headers: { "content-type": "application/json" } });

describe("/envelope", () => {
  it("accepts an opaque envelope and returns an id", async () => {
    const r = await post("/envelope", { relay_mailbox: "m1", sender_key_id: "kid_x", ciphertext: "deadbeef" });
    expect(r.status).toBe(200);
    expect((await r.json<{ id: string }>()).id).toMatch(/^env_/);
  });
  it("rejects an oversized envelope with 413", async () => {
    const r = await post("/envelope", { relay_mailbox: "m2", ciphertext: "z".repeat(70_000) });
    expect(r.status).toBe(413);
  });
});
