/**
 * Bridge client — talks to the companion-emergence bridge daemon.
 *
 * In Tauri builds, calls a Rust command to read the persona's
 * bridge.json (port + auth_token). In Vite dev, falls back to env
 * vars VITE_BRIDGE_URL + VITE_BRIDGE_TOKEN so the app still runs in
 * a plain browser tab.
 */

import { invoke } from "@tauri-apps/api/core";

export interface BridgeCredentials {
  url: string;
  authToken: string | null;
}

/** Cache after first resolution — bridge port doesn't change mid-session. */
let cached: BridgeCredentials | null = null;

export async function getBridgeCredentials(persona = "nell"): Promise<BridgeCredentials> {
  if (cached) return cached;

  // Try Tauri command first (production build path)
  try {
    const creds = await invoke<{ port: number; auth_token: string | null }>(
      "get_bridge_credentials",
      { persona },
    );
    cached = {
      url: `http://127.0.0.1:${creds.port}`,
      authToken: creds.auth_token,
    };
    return cached;
  } catch {
    // Browser dev fallback — read from import.meta.env
    const url = (import.meta.env.VITE_BRIDGE_URL as string) ?? "http://127.0.0.1:50000";
    const token = (import.meta.env.VITE_BRIDGE_TOKEN as string) ?? null;
    cached = { url, authToken: token };
    return cached;
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

export async function fetchPersonaState(): Promise<PersonaState> {
  const creds = await getBridgeCredentials();
  const r = await fetch(`${creds.url}/persona/state`, { headers: authHeaders(creds) });
  if (!r.ok) throw new Error(`/persona/state ${r.status}`);
  return await r.json();
}

export interface NewSessionResponse {
  session_id: string;
  persona: string;
  created_at: string;
}

export async function newSession(): Promise<string> {
  const creds = await getBridgeCredentials();
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

export async function sendChat(sessionId: string, message: string): Promise<ChatResponse> {
  const creds = await getBridgeCredentials();
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

export async function closeSession(sessionId: string): Promise<void> {
  const creds = await getBridgeCredentials();
  await fetch(`${creds.url}/sessions/close`, {
    method: "POST",
    headers: authHeaders(creds),
    body: JSON.stringify({ session_id: sessionId }),
  });
}
