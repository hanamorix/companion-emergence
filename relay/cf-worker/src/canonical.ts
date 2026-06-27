// Byte-identical to brain/kindled_link/codec.canonical_json:
//   json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
export function canonicalJson(value: unknown): string {
  return ser(value);
}

function ser(v: unknown): string {
  if (v === null) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : JSON.stringify(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return JSON.stringify(v); // JS escapes non-ascii minimally, matching ensure_ascii=False for ASCII hex/base64 payloads; parity test pins the auth-body shape
  if (Array.isArray(v)) return "[" + v.map(ser).join(",") + "]";
  if (typeof v === "object") {
    const keys = Object.keys(v as Record<string, unknown>).sort();
    return (
      "{" +
      keys.map((k) => JSON.stringify(k) + ":" + ser((v as Record<string, unknown>)[k])).join(",") +
      "}"
    );
  }
  throw new Error("uncanonicalisable value");
}
