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

function hasTauriRuntime(): boolean {
  const w = typeof window === "undefined" ? undefined : window as unknown as Record<string, unknown>;
  return Boolean(w && ("__TAURI_INTERNALS__" in w || "__TAURI__" in w));
}

export async function getBridgeCredentials(persona: string): Promise<BridgeCredentials> {
  const hit = cache.get(persona);
  if (hit) return hit;

  // Try Tauri command first (production build path).
  try {
    const creds = await invoke<{ port: number; auth_token: string | null }>(
      "get_bridge_credentials",
      { persona },
    );
    const result: BridgeCredentials = {
      url: `http://tauri.localhost:${creds.port}`,
      port: creds.port,
      authToken: creds.auth_token,
    };
    cache.set(persona, result);
    return result;
  } catch (e) {
    // Browser dev fallback only. In packaged Tauri production, silently
    // guessing 50000 can connect to the wrong persona or another local
    // service; surface the real credential failure instead.
    if (hasTauriRuntime()) throw e;

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

function authOnlyHeaders(creds: BridgeCredentials): HeadersInit {
  return creds.authToken ? { Authorization: `Bearer ${creds.authToken}` } : {};
}

async function bridgeFetch(
  persona: string,
  makeRequest: (creds: BridgeCredentials) => Promise<Response>,
): Promise<Response> {
  const creds = await getBridgeCredentials(persona);
  try {
    const response = await makeRequest(creds);
    if (response.status !== 401 && response.status !== 403) return response;
  } catch {
    // Network failures usually mean the bridge restarted and rewrote bridge.json.
    // Fall through to the single credential refresh below.
  }

  resetBridgeCredentialCache(persona);
  const fresh = await getBridgeCredentials(persona);
  return await makeRequest(fresh);
}

export interface InteriorEntry {
  summary: string;
  /** ISO-8601 timestamp the entry was written at, or null if the
   *  writer didn't record one (legacy entries pre-2026-05-08 may
   *  not have it). */
  ts: string | null;
}

export interface PersonaState {
  persona: string;
  emotions: Record<string, number>;
  body: BodyState | null;
  interior: {
    dream: InteriorEntry | null;
    research: InteriorEntry | null;
    heartbeat: InteriorEntry | null;
    reflex: InteriorEntry | null;
  };
  soul_highlight: SoulHighlight | null;
  connection: {
    provider: string | null;
    model: string | null;
    last_heartbeat_at: string | null;
  };
  mode: "live" | "bridge_down" | "provider_down" | "offline";
  /** True iff orphan session buffers from a previous shutdown are still
   *  being re-ingested. Drives the chat panel "reconnecting your previous
   *  chat" banner. Optional in the type so older bridge builds (pre-Phase
   *  3.A) without the field still parse — falls through to undefined and
   *  the UI treats it as false. */
  recovering?: boolean;
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
  const r = await bridgeFetch(persona, (creds) =>
    fetch(`${creds.url}/persona/state`, { headers: authHeaders(creds) }),
  );
  if (!r.ok) throw new Error(`/persona/state ${r.status}`);
  return await r.json();
}

export async function fetchPersonaState(persona: string): Promise<PersonaState> {
  return await fetchPersonaStateOnce(persona);
}

export interface NewSessionResponse {
  session_id: string;
  persona: string;
  created_at: string;
}

export async function newSession(persona: string): Promise<string> {
  const r = await bridgeFetch(persona, (creds) => fetch(`${creds.url}/session/new`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ client: "tauri" }),
  }));
  if (!r.ok) throw new Error(`/session/new ${r.status}`);
  const data = (await r.json()) as NewSessionResponse;
  return data.session_id;
}

/**
 * Ask the bridge for the persona's currently-active session, if any.
 *
 * Returns a session id when the bridge has a recent open session
 * (younger than the finalize threshold) we should reattach to, or
 * ``null`` when there's nothing to attach to and the caller should
 * fall back to ``newSession``. Throws on non-2xx so the caller can
 * decide whether to swallow the failure or surface it; ChatPanel
 * treats a throw the same as null (transient network flake).
 *
 * Part of the F-201 sticky-session reattach: combined with the
 * server-side hydration path and the non-destructive shutdown drain,
 * this lets Cmd-Q + reopen pick up mid-thread instead of starting a
 * fresh session every cold start.
 */
export async function fetchActiveSession(persona: string): Promise<string | null> {
  const r = await bridgeFetch(persona, (creds) =>
    fetch(`${creds.url}/sessions/active`, { headers: authHeaders(creds) }),
  );
  if (!r.ok) throw new Error(`/sessions/active ${r.status}`);
  const data = (await r.json()) as { session_id: string | null };
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
  const r = await bridgeFetch(persona, (creds) => fetch(`${creds.url}/chat`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ session_id: sessionId, message }),
  }));
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
  const fd = new FormData();
  fd.append("file", file);
  // Note: don't set Content-Type — fetch will pick the multipart boundary.
  const r = await bridgeFetch(persona, (creds) =>
    fetch(`${creds.url}/upload`, { method: "POST", headers: authOnlyHeaders(creds), body: fd }),
  );
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
  const r = await bridgeFetch(persona, (creds) => fetch(`${creds.url}/sessions/close`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ session_id: sessionId }),
    keepalive: options.keepalive,
  }));
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
