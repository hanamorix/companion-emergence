export function errString(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  if (e === null) return "null";
  if (e === undefined) return "undefined";
  if (typeof e === "object") {
    try { return JSON.stringify(e); } catch { return String(e); }
  }
  return String(e);
}
