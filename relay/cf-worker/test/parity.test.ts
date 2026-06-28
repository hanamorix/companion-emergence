import { describe, it, expect } from "vitest";
import { canonicalJson } from "../src/canonical";
import { verifyEd25519 } from "../src/crypto";

describe("canonical JSON parity with Python codec", () => {
  it("sorts keys, no spaces, preserves non-ascii", () => {
    expect(canonicalJson({ b: 1, a: 2 })).toBe('{"a":2,"b":1}');
    expect(canonicalJson({ purpose: "kindled-relay-auth/1", mailbox: "m1", nonce: "n1" }))
      .toBe('{"mailbox":"m1","nonce":"n1","purpose":"kindled-relay-auth/1"}');
    expect(canonicalJson({ s: "café" })).toBe('{"s":"café"}');
  });
});

describe("Ed25519 verify", () => {
  it("accepts a valid signature and rejects a tampered one", async () => {
    // KAT vector — RFC 8032 Ed25519 test vector 1: seed = 9d61b19d...
    // Confirmed via Python:
    //   from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as K
    //   k = K.from_private_bytes(bytes.fromhex('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60'))
    //   print(k.public_key().public_bytes_raw().hex())  # d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a
    //   print(k.sign(b'').hex())                        # e5564300...
    // Note: brief had wrong pub key (3b6a27b...) — corrected to Python-confirmed value.
    const pub = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const sig =
      "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b";
    const msg = new TextEncoder().encode("");
    expect(await verifyEd25519(pub, sig, msg)).toBe(true);
    expect(await verifyEd25519(pub, sig, new TextEncoder().encode("x"))).toBe(false);
    expect(await verifyEd25519("zz", sig, msg)).toBe(false); // bad hex → false, no throw
  });
});
