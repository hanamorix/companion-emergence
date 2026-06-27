import { env, SELF } from "cloudflare:test";
import { describe, it, expect, beforeEach } from "vitest";
import { applySchema } from "./apply-schema";

beforeEach(async () => { await applySchema(env.DB); });
const post = (path: string, body: unknown) =>
  SELF.fetch("https://relay.test" + path, { method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json" } });

describe("register + challenge", () => {
  it("register returns owner; challenge 404s until registered then issues a nonce", async () => {
    let r = await post("/mailbox/challenge", { mailbox_id: "m1" });
    expect(r.status).toBe(404);
    r = await post("/mailbox/register", { mailbox_id: "m1", identity_pub: "pubA" });
    expect(await r.json()).toEqual({ ok: true, owner: "pubA" });
    r = await post("/mailbox/challenge", { mailbox_id: "m1" });
    expect(r.status).toBe(200);
    const { nonce } = await r.json<{ nonce: string }>();
    expect(nonce).toMatch(/^[0-9a-f]{32}$/);
  });
});
