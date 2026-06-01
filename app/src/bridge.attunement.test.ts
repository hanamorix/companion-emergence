// Vitest coverage for fetchAttunement — Task 20 of the v0.0.28-alpha.1
// attunement work. Backs the AttunementPanel data path: current read,
// learned patterns, and backfill summary from GET /persona/attunement.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Stub Tauri invoke so getBridgeCredentials can resolve without a real
// Tauri runtime under jsdom — same pattern as bridge.history.test.ts.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd !== "get_bridge_credentials") throw new Error(`unexpected cmd ${cmd}`);
    if (args.persona === "nell") {
      return { port: 41001, auth_token: "nell-tok" };
    }
    throw new Error(`unknown persona ${args.persona}`);
  }),
}));

import { fetchAttunement, resetBridgeCredentialCache } from "./bridge";

describe("fetchAttunement", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("calls /persona/attunement with bearer auth and parses payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          current_read: {
            ts: "2026-05-31T12:00:00Z",
            source_turn_id: "t1",
            tone_label: "warm",
            tone_justification: "soft phrasing",
            cadence_label: "measured",
            cadence_justification: "full sentences",
            mood_valence: 0.3,
            mood_intensity: 0.5,
            predicted_arc_shape: "settling in",
            schema_version: "0.0.28-alpha.1",
          },
          learned_patterns: [],
          backfill: null,
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchAttunement("nell");
    expect(result.current_read?.tone_label).toBe("warm");
    expect(result.learned_patterns).toEqual([]);
    expect(result.backfill).toBeNull();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0]![0] as string;
    expect(url).toContain("/persona/attunement");
    const opts = fetchMock.mock.calls[0]![1] as RequestInit;
    expect((opts.headers as Record<string, string>)["Authorization"]).toBe("Bearer nell-tok");
  });

  it("returns nulls and empty arrays for a fresh persona", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ current_read: null, learned_patterns: [], backfill: null }),
          { status: 200 },
        ),
      ),
    );

    const result = await fetchAttunement("nell");
    expect(result.current_read).toBeNull();
    expect(result.learned_patterns).toEqual([]);
    expect(result.backfill).toBeNull();
  });

  it("returns learned patterns with all fields intact", async () => {
    const pattern = {
      id: "lp-abc",
      category: "tone",
      canonical_key: "prefers_direct",
      description: "Uses short, direct sentences",
      evidence_count: 5,
      maturity: "forming",
      first_seen_at: "2026-05-01T00:00:00Z",
      last_confirmed_at: "2026-05-30T00:00:00Z",
      last_addressed_at: null,
      crystallised_at: null,
      falsified_at: null,
      examples: ["just do it", "no fluff"],
      schema_version: "0.0.28-alpha.1",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ current_read: null, learned_patterns: [pattern], backfill: null }),
          { status: 200 },
        ),
      ),
    );

    const result = await fetchAttunement("nell");
    expect(result.learned_patterns).toHaveLength(1);
    expect(result.learned_patterns[0].canonical_key).toBe("prefers_direct");
    expect(result.learned_patterns[0].maturity).toBe("forming");
    expect(result.learned_patterns[0].examples).toEqual(["just do it", "no fluff"]);
  });

  it("throws on non-OK response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("unauthorized", { status: 401 })),
    );

    await expect(fetchAttunement("nell")).rejects.toThrow(/401/);
  });
});
