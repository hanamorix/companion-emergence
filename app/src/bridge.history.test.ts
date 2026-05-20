// Vitest coverage for fetchChatHistory — Phase 3B of the v0.0.15-alpha.2
// chat-reliability work. Backs the mount-time hydration path in ChatPanel
// so the bridge's on-disk turn log surfaces in the UI on reopen.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the Tauri invoke so getBridgeCredentials can resolve without a
// real Tauri runtime under jsdom. Match the persona-scoped pattern from
// bridge.test.ts so behaviour stays consistent.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd !== "get_bridge_credentials") throw new Error(`unexpected cmd ${cmd}`);
    if (args.persona === "nell") {
      return { port: 41001, auth_token: "nell-tok" };
    }
    throw new Error(`unknown persona ${args.persona}`);
  }),
}));

import { fetchChatHistory, resetBridgeCredentialCache } from "./bridge";

describe("fetchChatHistory", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits /chat/history with session_id and limit", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ messages: [], next_before_turn: null }), {
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchChatHistory("nell", "s_a", 200);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/chat/history");
    expect(url).toContain("session_id=s_a");
    expect(url).toContain("limit=200");
  });

  it("returns the parsed messages array", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          messages: [
            { role: "user", content: "hi", turn: 1, ts: "2026-05-20T10:00:00Z" },
            {
              role: "assistant",
              content: "hello",
              turn: 2,
              ts: "2026-05-20T10:00:05Z",
            },
          ],
          next_before_turn: null,
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const res = await fetchChatHistory("nell", "s_a", 200);
    expect(res.messages).toHaveLength(2);
    expect(res.messages[0].content).toBe("hi");
    expect(res.messages[0].role).toBe("user");
    expect(res.messages[1].content).toBe("hello");
    expect(res.next_before_turn).toBeNull();
  });

  it("passes before_turn when supplied (pagination cursor)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ messages: [], next_before_turn: null }), {
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchChatHistory("nell", "s_a", 50, 42);

    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("before_turn=42");
    expect(url).toContain("limit=50");
  });

  it("throws on non-2xx so the caller can fall back to empty UI state", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("nope", { status: 500 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchChatHistory("nell", "s_a")).rejects.toThrow(/chat\/history/);
  });
});
