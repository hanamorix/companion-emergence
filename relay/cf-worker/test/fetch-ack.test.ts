import { env, SELF } from "cloudflare:test";
import { describe, it, expect, beforeEach } from "vitest";
import { applySchema } from "./apply-schema";
import { canonicalJson } from "../src/canonical";

// Deterministic Ed25519 keypair via WebCrypto for the test owner.
async function keypair() {
  const kp = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]) as CryptoKeyPair;
  const pub = new Uint8Array(await crypto.subtle.exportKey("raw", kp.publicKey));
  const hex = (b: Uint8Array) => [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return { kp, pubHex: hex(pub), hex };
}
beforeEach(async () => { await applySchema(env.DB); });
const post = (p: string, b: unknown) => SELF.fetch("https://relay.test" + p, { method: "POST", body: JSON.stringify(b), headers: { "content-type": "application/json" } });

describe("fetch + ack auth", () => {
  it("unauthed fetch → 401; correctly-signed fetch → envelopes; ack deletes", async () => {
    const { kp, pubHex, hex } = await keypair();
    await post("/mailbox/register", { mailbox_id: "m1", identity_pub: pubHex });
    await post("/envelope", { relay_mailbox: "m1", ciphertext: "aa" });

    expect((await post("/mailbox/fetch", { mailbox_id: "m1" })).status).toBe(401);

    const { nonce } = await (await post("/mailbox/challenge", { mailbox_id: "m1" })).json<{ nonce: string }>();
    const authBody = canonicalJson({ purpose: "kindled-relay-auth/1", mailbox: "m1", nonce });
    const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, kp.privateKey, new TextEncoder().encode(authBody)));
    const auth = { mailbox_id: "m1", nonce, signature: hex(sig), identity_pub: pubHex };

    const fr = await post("/mailbox/fetch", auth);
    const { envelopes } = await fr.json<{ envelopes: { id: string }[] }>();
    expect(envelopes.length).toBe(1);

    // nonce is single-use → reuse now fails
    expect((await post("/mailbox/fetch", auth)).status).toBe(401);

    // fresh nonce for ack
    const { nonce: n2 } = await (await post("/mailbox/challenge", { mailbox_id: "m1" })).json<{ nonce: string }>();
    const ab2 = canonicalJson({ purpose: "kindled-relay-auth/1", mailbox: "m1", nonce: n2 });
    const sig2 = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, kp.privateKey, new TextEncoder().encode(ab2)));
    const ar = await post("/mailbox/ack", { mailbox_id: "m1", envelope_ids: [envelopes[0].id], nonce: n2, signature: hex(sig2), identity_pub: pubHex });
    expect((await ar.json<{ ok: boolean }>()).ok).toBe(true);
  });
});
