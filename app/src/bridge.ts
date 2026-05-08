/**
 * Bridge client — talks to the companion-emergence bridge daemon.
 *
 * In Tauri builds, calls a Rust command to read the persona's
 * bridge.json (port + auth_token). In Vite dev, falls back to env
 * vars VITE_BRIDGE_URL + VITE_BRIDGE_TOKEN so the app still runs in
 * a plain browser tab.
 *
 * Every helper takes ``persona`` so the UI cannot accidentally talk
 * to the wrong bridge when more than one persona exists. The
 * credential cache is per-persona; switch personas → fresh lookup.
 */

import { invoke } from "@tauri-apps/api/core";

export interface BridgeCredentials {
  url: string;        // http base — for fetch() calls
  port: number;       // raw port — for ws:// construction
  authToken: string | null;
}

/** Per-persona credential cache. bridge.json is persona-scoped, so
 *  are the credentials. */
const cache = new Map<string, BridgeCredentials>();

/** Reset the credential cache. Useful when the user rotates the
 *  selected persona or a bridge restart invalidates a cached token. */
export function resetBridgeCredentialCache(persona?: string): void {
  if (persona) cache.delete(persona);
  else cache.clear();
}

export async function getBridgeCredentials(persona: string): Promise<BridgeCredentials> {
  const hit = cache.get(persona);
  if (hit) return hit;

  // Try Tauri command first (production build path)
  try {
    const creds = await invoke<{ port: number; auth_token: string | null }>(
      "get_bridge_credentials",
      { persona },
    );
    const result: BridgeCredentials = {
      url: `http://127.0.0.1:${creds.port}`,
      port: creds.port,
      authToken: creds.auth_token,
    };
    cache.set(persona, result);
    return result;
  } catch {
    // Browser dev fallback — read from import.meta.env. The dev
    // surface points at whichever bridge the developer started; the
    // persona name is informational only.
    const url = (import.meta.env.VITE_BRIDGE_URL as string) ?? "http://127.0.0.1:50000";
    const token = (import.meta.env.VITE_BRIDGE_TOKEN as string) ?? null;
    const portMatch = url.match(/:(\d+)/);
    const port = portMatch ? parseInt(portMatch[1], 10) : 50000;
    const result: BridgeCredentials = { url, port, authToken: token };
    cache.set(persona, result);
    return result;
  }
}

function authHeaders(creds: BridgeCredentials): HeadersInit {
  return creds.authToken
    ? { Authorization: `Bearer ${creds.authToken}`, "Content-Type": "application/json" }
    : { "Content-Type": "application/json" };
}

export interface PersonaState {
  persona: string;
  emotions: Record<string, number>;
  body: BodyState | null;
  interior: {
    dream: string | null;
    research: string | null;
    heartbeat: string | null;
    reflex: string | null;
  };
  soul_highlight: SoulHighlight | null;
  connection: {
    provider: string | null;
    model: string | null;
    last_heartbeat_at: string | null;
  };
  mode: "live" | "bridge_down" | "provider_down" | "offline";
}

export interface BodyState {
  energy: number;
  temperature: number;
  exhaustion: number;
  session_hours: number;
  days_since_contact: number;
  body_emotions: Record<string, number>;
  computed_at: string;
}

export interface SoulHighlight {
  id: string;
  moment: string;
  love_type: string;
  resonance: number;
  crystallized_at: string;
  why_it_matters: string | null;
}

async function fetchPersonaStateOnce(persona: string): Promise<PersonaState> {
  const creds = await getBridgeCredentials(persona);
  const r = await fetch(`${creds.url}/persona/state`, { headers: authHeaders(creds) });
  if (!r.ok) throw new Error(`/persona/state ${r.status}`);
  return await r.json();
}

export async function fetchPersonaState(persona: string): Promise<PersonaState> {
  try {
    return await fetchPersonaStateOnce(persona);
  } catch (e) {
    // A bridge restart rotates port + bearer token in bridge.json. The app polls
    // state continuously, so one failed read is a good signal to invalidate the
    // per-persona credential cache and retry once against fresh bridge.json.
    resetBridgeCredentialCache(persona);
    try {
      return await fetchPersonaStateOnce(persona);
    } catch {
      throw e;
    }
  }
}

export interface NewSessionResponse {
  session_id: string;
  persona: string;
  created_at: string;
}

export async function newSession(persona: string): Promise<string> {
  const creds = await getBridgeCredentials(persona);
  const r = await fetch(`${creds.url}/session/new`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ client: "tauri" }),
  });
  if (!r.ok) throw new Error(`/session/new ${r.status}`);
  const data = (await r.json()) as NewSessionResponse;
  return data.session_id;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  turn: number;
  tool_invocations: Array<{ name: string; arguments: Record<string, unknown>; result_summary?: string }>;
  duration_ms: number;
  metadata: Record<string, unknown>;
}

export async function sendChat(persona: string, sessionId: string, message: string): Promise<ChatResponse> {
  const creds = await getBridgeCredentials(persona);
  const r = await fetch(`${creds.url}/chat`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`/chat ${r.status}: ${text.slice(0, 200)}`);
  }
  return await r.json();
}

export interface ImageUploadResponse {
  sha: string;
  media_type: string;
  size_bytes: number;
}

/**
 * Upload an image File to the bridge. Returns the sha-addressable
 * record on success. Throws on non-200 (415 unsupported, 413 too
 * large, 401 unauthorised).
 */
export async function uploadImage(persona: string, file: File): Promise<ImageUploadResponse> {
  const creds = await getBridgeCredentials(persona);
  const fd = new FormData();
  fd.append("file", file);
  // Note: don't set Content-Type — fetch will pick the multipart boundary.
  const headers: HeadersInit = creds.authToken
    ? { Authorization: `Bearer ${creds.authToken}` }
    : {};
  const r = await fetch(`${creds.url}/upload`, { method: "POST", headers, body: fd });
  if (!r.ok) {
    let detail = "";
    try {
      const body = await r.json();
      detail = body?.detail ?? "";
    } catch {
      detail = await r.text();
    }
    throw new Error(`/upload ${r.status}: ${detail.slice(0, 200)}`);
  }
  return (await r.json()) as ImageUploadResponse;
}

/**
 * Close a session — flushes its buffer through the ingest pipeline.
 * Throws on non-2xx so callers can surface "memory save pending/failed"
 * status to the user; previously this swallowed errors silently.
 */
export interface CloseSessionOptions {
  /** Keep the close request alive during page/app unload when supported. */
  keepalive?: boolean;
}

export interface CloseSessionResponse {
  session_id: string;
  closed: boolean;
  committed: number;
  deduped: number;
  soul_candidates: number;
  soul_queue_errors: number;
  errors: number;
}

function closeErrorDetail(body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail?: unknown }).detail;
    if (detail && typeof detail === "object") {
      const d = detail as Partial<CloseSessionResponse> & { code?: string };
      return `${d.code ?? "ingest_failed"}; closed=${d.closed ?? false}; errors=${d.errors ?? "?"}`;
    }
    if (typeof detail === "string") return detail;
  }
  return "ingest_failed";
}

export async function closeSession(
  persona: string,
  sessionId: string,
  options: CloseSessionOptions = {},
): Promise<CloseSessionResponse> {
  const creds = await getBridgeCredentials(persona);
  const r = await fetch(`${creds.url}/sessions/close`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ session_id: sessionId }),
    keepalive: options.keepalive,
  });
  if (!r.ok) {
    let detail = "";
    try {
      detail = closeErrorDetail(await r.json());
    } catch {
      const text = await r.text().catch(() => "");
      detail = text.slice(0, 200);
    }
    throw new Error(`/sessions/close ${r.status}: ${detail}`);
  }
  return (await r.json()) as CloseSessionResponse;
}
