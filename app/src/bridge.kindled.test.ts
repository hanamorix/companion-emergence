// Vitest coverage for kindled-link bridge fns — Phase 6 Task 6.
// Tests the fetch/control functions for Kindled Links panel.
// Mirrors the exact mock harness from bridge.attunement.test.ts.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Stub Tauri invoke so getBridgeCredentials can resolve without a real
// Tauri runtime under jsdom — same pattern as bridge.attunement.test.ts.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd !== "get_bridge_credentials") throw new Error(`unexpected cmd ${cmd}`);
    if (args.persona === "nell") {
      return { port: 41001, auth_token: "nell-tok" };
    }
    throw new Error(`unknown persona ${args.persona}`);
  }),
}));

import {
  fetchKindledPeers,
  fetchKindledTranscript,
  fetchKindledHolds,
  createKindledInvite,
  acceptKindledInvite,
  setKindledConsent,
  fetchKindledMyCode,
  connectKindled,
  runKindledSelfTest,
  resetBridgeCredentialCache,
} from "./bridge";

describe("kindled-link bridge", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetchKindledPeers calls /kindled-link/peers with bearer auth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          peers: [
            {
              peer_id: "kid_a",
              fingerprint: "abc123",
              relay_url: "https://relay.example.com",
              consent_state: "active",
              stage: "connected",
              affinity_tags: ["trusted"],
              has_active_session: true,
            },
          ],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchKindledPeers("nell");
    expect(result).toHaveLength(1);
    expect(result[0].peer_id).toBe("kid_a");
    expect(result[0].consent_state).toBe("active");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/peers");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
  });

  it("fetchKindledTranscript calls /kindled-link/peers/{id}/transcript with bearer auth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          transcript: [
            { seq: 1, direction: "outbound", text: "hello", provenance: "nell", ts: "2026-06-21T10:00:00Z" },
            { seq: 2, direction: "inbound", text: "hi", provenance: "peer", ts: "2026-06-21T10:01:00Z" },
          ],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchKindledTranscript("nell", "kid_a");
    expect(result).toHaveLength(2);
    expect(result[0].text).toBe("hello");
    expect(result[1].direction).toBe("inbound");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/peers/kid_a/transcript");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
  });

  it("fetchKindledHolds calls /kindled-link/holds with bearer auth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          held_count: 2,
          items: [
            { session_id: "sess_x", created_at: "2026-06-20T08:00:00Z" },
            { session_id: "sess_y", created_at: "2026-06-19T14:00:00Z" },
          ],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchKindledHolds("nell");
    expect(result.held_count).toBe(2);
    expect(result.items).toHaveLength(2);

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/holds");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
  });

  it("createKindledInvite POSTs to /kindled-link/invite with relay_url", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          invite: { code: "inv_123", expires_at: "2026-06-28T10:00:00Z" },
          fingerprint: "fpr_abc",
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await createKindledInvite("nell", "https://relay.example.com");
    expect(result.fingerprint).toBe("fpr_abc");
    expect(result.invite).toBeDefined();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/invite");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(opts.method).toBe("POST");
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
    const body = JSON.parse(opts.body as string);
    expect(body.relay_url).toBe("https://relay.example.com");
  });

  it("acceptKindledInvite POSTs to /kindled-link/invite/accept", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          peer_id: "kid_new",
          fingerprint_phrase: "apple banana cherry",
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const invite = { code: "inv_123", data: "..." };
    const result = await acceptKindledInvite("nell", invite);
    expect(result.peer_id).toBe("kid_new");
    expect(result.fingerprint_phrase).toBe("apple banana cherry");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/invite/accept");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(opts.method).toBe("POST");
    const body = JSON.parse(opts.body as string);
    expect(body.invite).toEqual(invite);
  });

  it("setKindledConsent POSTs action to /kindled-link/peers/{id}/consent", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await setKindledConsent("nell", "kid_a", "pause");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/peers/kid_a/consent");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(opts.method).toBe("POST");
    const body = JSON.parse(opts.body as string);
    expect(body.action).toBe("pause");
  });

  it("fetchKindledMyCode GETs /kindled-link/my-code with bearer auth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ code: "kindled1:abc", fingerprint_phrase: "aa bb" }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchKindledMyCode("nell");
    expect(result.code).toBe("kindled1:abc");
    expect(result.fingerprint_phrase).toBe("aa bb");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/my-code");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect((opts.method ?? "GET")).toBe("GET");
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
  });

  it("connectKindled POSTs /kindled-link/connect with the code", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          peer_id: "kid_x",
          consent_state: "paired",
          relay_url: "https://r",
          fingerprint_phrase: "aa",
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await connectKindled("nell", "kindled1:abc");
    expect(result.consent_state).toBe("paired");
    expect(result.peer_id).toBe("kid_x");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/connect");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(opts.method).toBe("POST");
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
    const body = JSON.parse(opts.body as string);
    expect(body.code).toBe("kindled1:abc");
  });

  it("connectKindled surfaces server detail on error", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ detail: "code expired" }),
        { status: 400 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(connectKindled("nell", "kindled1:bad")).rejects.toThrow("code expired");
  });

  it("runKindledSelfTest POSTs /kindled-link/self-test and returns stages", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ok: true,
          relay_url: "https://r",
          stages: [{ name: "relay_reachable", ok: true, detail: "" }],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await runKindledSelfTest("nell");
    expect(result.ok).toBe(true);
    expect(result.stages[0].name).toBe("relay_reachable");
    expect(result.relay_url).toBe("https://r");

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/kindled-link/self-test");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(opts.method).toBe("POST");
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
  });
});
