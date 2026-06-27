import { env, SELF } from "cloudflare:test";
import { describe, it, expect } from "vitest";

describe("worker skeleton", () => {
  it("responds to /healthz", async () => {
    const res = await SELF.fetch("https://relay.test/healthz");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
  });
});
