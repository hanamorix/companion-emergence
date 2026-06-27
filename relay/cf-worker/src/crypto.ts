function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0 || /[^0-9a-fA-F]/.test(hex)) throw new Error("bad hex");
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

export async function verifyEd25519(
  pubHex: string,
  sigHex: string,
  message: Uint8Array,
): Promise<boolean> {
  try {
    const pub = hexToBytes(pubHex);
    const sig = hexToBytes(sigHex);
    const key = await crypto.subtle.importKey("raw", pub, { name: "Ed25519" }, false, ["verify"]);
    return await crypto.subtle.verify({ name: "Ed25519" }, key, sig, message);
  } catch {
    return false; // fail-soft, matches dev_relay _verify
  }
}
