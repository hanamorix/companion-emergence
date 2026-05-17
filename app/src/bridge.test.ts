// Vitest smoke tests for bridge.ts — pins the audit-2026-05-07 P1-2
// fix so the persona scoping + per-persona cache can't regress silently.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the Tauri invoke so we can drive get_bridge_credentials
// without a real Tauri runtime under jsdom. Two distinct personas →
// two distinct credential payloads, asserted both ways.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd !== "get_bridge_credentials") throw new Error(`unexpected cmd ${cmd}`);
    if (args.persona === "alice") {
      return { port: 41001, auth_token: "alice-tok" };
    }
    if (args.persona === "bob") {
      return { port: 42002, auth_token: "bob-tok" };
    }
    throw new Error(`unknown persona ${args.persona}`);
  }),
}));

import { invoke } from "@tauri-apps/api/core";
import {
  closeSession,
  fetchPersonaFeed,
  fetchPersonaState,
  getBridgeCredentials,
  newSession,
  resetBridgeCredentialCache,
  uploadImage,
} from "./bridge";

describe("getBridgeCredentials", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("invokes get_bridge_credentials with the supplied persona", async () => {
    const creds = await getBridgeCredentials("alice");
    expect(creds.port).toBe(41001);
    expect(creds.authToken).toBe("alice-tok");
    expect(creds.url).toBe("http://127.0.0.1:41001");
    expect(invoke).toHaveBeenCalledWith("get_bridge_credentials", {
      persona: "alice",
    });
  });

  it("does not fall back to the default dev bridge when a Tauri runtime is present", async () => {
    vi.stubGlobal("__TAURI_INTERNALS__", {});
    await expect(getBridgeCredentials("unknown")).rejects.toThrow("unknown persona unknown");
  });

  it("does use the default bridge fallback in plain browser dev mode", async () => {
    const creds = await getBridgeCredentials("unknown");
    expect(creds).toMatchObject({
      url: "http://127.0.0.1:50000",
      port: 50000,
      authToken: null,
    });
  });

  it("caches per-persona — second call same persona doesn't re-invoke", async () => {
    await getBridgeCredentials("alice");
    await getBridgeCredentials("alice");
    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it("does not bleed credentials across personas", async () => {
    const a = await getBridgeCredentials("alice");
    const b = await getBridgeCredentials("bob");
    expect(a.port).toBe(41001);
    expect(b.port).toBe(42002);
    expect(a.authToken).not.toBe(b.authToken);
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it("resetBridgeCredentialCache(persona) only invalidates that persona", async () => {
    await getBridgeCredentials("alice");
    await getBridgeCredentials("bob");
    resetBridgeCredentialCache("alice");
    await getBridgeCredentials("alice"); // re-invokes
    await getBridgeCredentials("bob");   // still cached
    expect(invoke).toHaveBeenCalledTimes(3);
  });

  it("refreshes credentials and retries persona state after one Load failed", async () => {
    const state = {
      persona: "alice",
      emotions: {},
      body: null,
      interior: { dream: null, research: null, heartbeat: null, reflex: null },
      soul_highlight: null,
      connection: { provider: null, model: null, last_heartbeat_at: null },
      mode: "live",
    };
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("Load failed"))
      .mockResolvedValueOnce(new Response(JSON.stringify(state), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchPersonaState("alice")).resolves.toMatchObject({ persona: "alice" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it("refreshes credentials and retries session creation after auth failure", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response("unauthorized", { status: 401 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        session_id: "session-2",
        persona: "alice",
        created_at: "2026-05-08T00:00:00Z",
      }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(newSession("alice")).resolves.toBe("session-2");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it("can keep session-close alive during app unload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ session_id: "session-1", closed: true, committed: 0, deduped: 0, soul_candidates: 0, soul_queue_errors: 0, errors: 0 }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(closeSession("alice", "session-1", { keepalive: true })).resolves.toMatchObject({ closed: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:41001/sessions/close",
      expect.objectContaining({
        method: "POST",
        keepalive: true,
        body: JSON.stringify({ session_id: "session-1" }),
      }),
    );
  });

  it("surfaces retryable close-session failures with closed=false detail", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: {
            code: "ingest_failed",
            session_id: "session-1",
            closed: false,
            committed: 0,
            deduped: 0,
            soul_candidates: 0,
            soul_queue_errors: 0,
            errors: 1,
          },
        }),
        { status: 502, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(closeSession("alice", "session-1")).rejects.toThrow("closed=false; errors=1");
  });

  it("refreshes credentials and retries uploads after a network failure", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("connection refused"))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        sha: "a".repeat(64),
        media_type: "image/png",
        size_bytes: 12,
      }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const file = new File([new Uint8Array([1, 2, 3])], "tiny.png", { type: "image/png" });
    await expect(uploadImage("alice", file)).resolves.toMatchObject({ media_type: "image/png" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(invoke).toHaveBeenCalledTimes(2);
  });
});

describe("fetchPersonaFeed", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the entries array on a 200 response", async () => {
    const fakeResponse = {
      entries: [
        {
          type: "dream",
          ts: "2026-05-17T01:00:00+00:00",
          opener: "I dreamed",
          body: "a hallway",
          audit_id: null,
        },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify(fakeResponse), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const entries = await fetchPersonaFeed("alice");
    expect(entries).toEqual(fakeResponse.entries);
  });

  it("returns an empty array when entries are empty", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify({ entries: [] }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const entries = await fetchPersonaFeed("alice");
    expect(entries).toEqual([]);
  });

  it("throws on a 401 response (auth failure)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("unauthorized", { status: 401 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchPersonaFeed("alice")).rejects.toThrow();
  });
});
