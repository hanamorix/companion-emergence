import { env, SELF } from "cloudflare:test";
import { describe, it, expect, beforeEach } from "vitest";
import { applySchema } from "./apply-schema";

beforeEach(async () => { await applySchema(env.DB); });

describe("rate limiting", () => {
  it("fails open when no CF-Connecting-IP header is present", async () => {
    // No client IP in test → limiter must not block (fail-open).
    const r = await SELF.fetch("https://relay.test/mailbox/register", {
      method: "POST", body: JSON.stringify({ mailbox_id: "m1", identity_pub: "p" }),
      headers: { "content-type": "application/json" },
    });
    expect(r.status).toBe(200);
  });
});
