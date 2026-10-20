// Vitest smoke tests for bridge.ts — pins the audit-2026-05-07 P1-2
// fix so the persona scoping + per-persona cache can't regress silently.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the Tauri invoke so we can drive get_bridge_credentials
// without a real Tauri runtime under jsdom. Two distinct personas →
// two distinct credential payloads, asserted both ways.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd === "force_restart_bridge") return undefined;
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
  acceptVoiceEdit,
  approvePendingWrite,
  closeSession,
  declinePendingWrite,
  fetchPersonaFeed,
  fetchPersonaState,
  getBridgeCredentials,
  invokeForceRestart,
  newSession,
  rejectVoiceEdit,
  resetBridgeCredentialCache,
  sendChat,
  setPersonaModel,
  setPersonaModelFull,
  setPersonaPronouns,
  shutdownBridge,
  snapshotSession,
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

  // #365: a request that failed in flight may already have reached the
  // bridge, so a non-idempotent POST must not be sent a second time.
  it.each([
    ["sendChat", () => sendChat("alice", "session-1", "hi")],
    ["newSession", () => newSession("alice")],
    ["snapshotSession", () => snapshotSession("alice", "session-1")],
    ["shutdownBridge", () => shutdownBridge("alice")],
  ])("does not replay %s after a network failure at an unchanged address", async (_name, call) => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("connection reset"))
      .mockResolvedValue(new Response(JSON.stringify({ entries: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(call()).rejects.toThrow("connection reset");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // The stale credentials were dropped, so the next call re-reads them.
    await fetchPersonaFeed("alice");
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it("replays a chat POST when the re-read credentials name a new address (the old bridge is gone)", async () => {
    await getBridgeCredentials("alice"); // cached: port 41001
    vi.mocked(invoke).mockImplementationOnce(async () => ({ port: 41009, auth_token: "alice-tok-2" }));
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("connection refused"))
      .mockResolvedValueOnce(new Response(JSON.stringify({ reply: "hi", turn: 1 }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(sendChat("alice", "session-1", "hi")).resolves.toMatchObject({ reply: "hi" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1][0]).toBe("http://127.0.0.1:41009/chat");
  });

  it("replays a chat POST the bridge rejected with 401 (auth runs before the handler)", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response("invalid token", { status: 401 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ reply: "hi", turn: 1 }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(sendChat("alice", "session-1", "hi")).resolves.toMatchObject({ reply: "hi" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("drops cached credentials when the app force-restarts the bridge", async () => {
    await getBridgeCredentials("alice");
    await invokeForceRestart("alice");
    await getBridgeCredentials("alice");
    const credentialReads = vi.mocked(invoke).mock.calls.filter(([cmd]) => cmd === "get_bridge_credentials");
    expect(credentialReads).toHaveLength(2);
  });

  it("(#456 C17 m) the forced path passes skipHardExitRequest to the kill helper command", async () => {
    await invokeForceRestart("alice");
    await invokeForceRestart("alice", true);
    const calls = vi.mocked(invoke).mock.calls.filter(([cmd]) => cmd === "force_restart_bridge");
    expect(calls.map(([, args]) => args)).toEqual([
      { persona: "alice", skipHardExitRequest: false },
      { persona: "alice", skipHardExitRequest: true },
    ]);
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

describe("setPersonaModel", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to /persona/config/model with the supplied model", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await setPersonaModel("alice", "opus");

    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/persona/config/model");
    expect((init!.method)?.toUpperCase()).toBe("POST");
    expect(init!.body).toContain("opus");
  });

  it("resolves without a value on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "haiku" }), { status: 200 }),
    ));

    await expect(setPersonaModel("alice", "haiku")).resolves.toBeUndefined();
  });

  it("setPersonaModelFull also returns the tier fallbacks", async () => {
    const fb = [{ family: "haiku", wanted: "claude-haiku-5", using: "haiku" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true, fallbacks: fb }), { status: 200 }),
    ));
    await expect(setPersonaModelFull("alice", "claude-sonnet-5")).resolves.toEqual({ checked: true, fallbacks: fb, tiersUnchecked: [] });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true, tiers_unchecked: ["claude-haiku-5"] }), { status: 200 }),
    ));
    await expect(setPersonaModelFull("alice", "claude-sonnet-5")).resolves.toEqual({ checked: true, fallbacks: [], tiersUnchecked: ["claude-haiku-5"] });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }),
    ));
    await expect(setPersonaModelFull("alice", "opus")).resolves.toEqual({ checked: undefined, fallbacks: [], tiersUnchecked: [] });
  });

  it("resolves to the bridge's checked flag for a custom id", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true }), { status: 200 }),
    ));
    await expect(setPersonaModel("alice", "claude-sonnet-5")).resolves.toBe(true);

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: false }), { status: 200 }),
    ));
    await expect(setPersonaModel("alice", "claude-sonnet-5")).resolves.toBe(false);
  });

  it("throws on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: false, error: "unknown_model" }), { status: 400 }),
    ));

    await expect(setPersonaModel("alice", "sonnet")).rejects.toThrow("setPersonaModel failed: 400");
  });

  // #440: a full id is checked by the bridge before it is saved.
  it("sends a full model id as written", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await setPersonaModel("alice", "claude-sonnet-5");

    expect(JSON.parse(spy.mock.calls[0]![1]!.body as string)).toEqual({ model: "claude-sonnet-5" });
  });

  it("turns a refused model into the specific message naming it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ ok: false, error: "model_unavailable", model: "claude-sonet-5" }),
        { status: 400 },
      ),
    ));

    const err = await setPersonaModel("alice", "claude-sonet-5").catch((e: Error) => e);
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toContain('"claude-sonet-5"');
    expect((err as Error).message).toMatch(/doesn't recognise/i);
    expect((err as Error).message).not.toMatch(/setPersonaModel failed/);
  });

  it("does not replay a custom id after a network failure (a replay would run a second check)", async () => {
    const spy = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", spy);

    await expect(setPersonaModel("alice", "claude-sonnet-5")).rejects.toThrow("Failed to fetch");
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("still replays an alias after a network failure", async () => {
    const spy = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }));
    vi.stubGlobal("fetch", spy);

    await expect(setPersonaModel("alice", "opus")).resolves.toBeUndefined();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});

describe("setPersonaPronouns", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to /persona/config/pronouns with the supplied preset", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, pronouns: {} }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);
    await setPersonaPronouns("alice", "he/him");
    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/persona/config/pronouns");
    expect((init!.method)?.toUpperCase()).toBe("POST");
    expect(init!.body).toContain("he/him");
  });

  it("names the stale-bridge cause on 404 (route absent = pre-update bridge)", async () => {
    const spy = vi.fn().mockResolvedValue(new Response(null, { status: 404 }));
    vi.stubGlobal("fetch", spy);
    await expect(setPersonaPronouns("alice", "he/him")).rejects.toThrow(
      /older version.*restart/i,
    );
  });

  it("keeps the plain status message for non-404 failures", async () => {
    const spy = vi.fn().mockResolvedValue(new Response(null, { status: 500 }));
    vi.stubGlobal("fetch", spy);
    await expect(setPersonaPronouns("alice", "he/him")).rejects.toThrow(
      "setPersonaPronouns failed: 500",
    );
  });
});

describe("acceptVoiceEdit / rejectVoiceEdit", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("acceptVoiceEdit POSTs to /initiate/voice-edit/accept with audit_id", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await acceptVoiceEdit("alice", "ia_001");

    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/initiate/voice-edit/accept");
    expect((init!.method)?.toUpperCase()).toBe("POST");
    const body = JSON.parse(String(init!.body));
    expect(body).toEqual({ audit_id: "ia_001" });
  });

  it("acceptVoiceEdit includes with_edits when editedText is provided", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await acceptVoiceEdit("alice", "ia_002", "the edited new text");

    const [, init] = spy.mock.calls[0]!;
    const body = JSON.parse(String(init!.body));
    expect(body).toEqual({ audit_id: "ia_002", with_edits: "the edited new text" });
  });

  it("acceptVoiceEdit omits with_edits when editedText is null", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await acceptVoiceEdit("alice", "ia_003", null);

    const [, init] = spy.mock.calls[0]!;
    const body = JSON.parse(String(init!.body));
    expect(body).not.toHaveProperty("with_edits");
    expect(body.audit_id).toBe("ia_003");
  });

  it("acceptVoiceEdit throws on non-2xx", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response("bad request", { status: 400 }),
    ));
    await expect(acceptVoiceEdit("alice", "ia_bad")).rejects.toThrow("/initiate/voice-edit/accept 400");
  });

  it("rejectVoiceEdit POSTs to /initiate/voice-edit/reject with audit_id", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await rejectVoiceEdit("alice", "ia_rej");

    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/initiate/voice-edit/reject");
    expect((init!.method)?.toUpperCase()).toBe("POST");
    const body = JSON.parse(String(init!.body));
    expect(body).toEqual({ audit_id: "ia_rej" });
  });

  it("rejectVoiceEdit throws on non-2xx", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response("server error", { status: 500 }),
    ));
    await expect(rejectVoiceEdit("alice", "ia_bad")).rejects.toThrow("/initiate/voice-edit/reject 500");
  });

  it("approvePendingWrite POSTs to /persona/writes/{id}/approve with the bearer", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await approvePendingWrite("alice", "w_001");

    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/persona/writes/w_001/approve");
    expect((init!.method)?.toUpperCase()).toBe("POST");
    expect((init!.headers as Record<string, string>)?.Authorization).toBe("Bearer alice-tok");
  });

  it("declinePendingWrite POSTs to /persona/writes/{id}/decline with the bearer", async () => {
    const spy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", spy);

    await declinePendingWrite("alice", "w_002");

    const [url, init] = spy.mock.calls[0]!;
    expect(String(url)).toContain("/persona/writes/w_002/decline");
    expect((init!.method)?.toUpperCase()).toBe("POST");
    expect((init!.headers as Record<string, string>)?.Authorization).toBe("Bearer alice-tok");
  });

  it("snapshots a session without closing it", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ session_id: "session-1", closed: false, committed: 0, deduped: 0, soul_candidates: 0, soul_queue_errors: 0, errors: 0 }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const r = await snapshotSession("alice", "session-1", { keepalive: true });
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:41001/sessions/snapshot",
      expect.objectContaining({
        method: "POST",
        keepalive: true,
        body: JSON.stringify({ session_id: "session-1" }),
      }),
    );
    expect(r).toMatchObject({ closed: false });
  });
});
