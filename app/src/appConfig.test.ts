/**
 * appConfig.ts — brain-update invoke wrappers (#286 slice 4).
 *
 * Asserts each wrapper calls invoke() with the exact Rust command name and
 * args, and returns whatever invoke() resolves to.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

import { invoke } from "@tauri-apps/api/core";
import { HEALTH_WAIT_MS } from "./lib/requestTimeout";
import { applyBrainUpdate, checkBrainUpdate, revertBrain, rollbackBrain } from "./appConfig";

const mockInvoke = vi.mocked(invoke);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("checkBrainUpdate", () => {
  it('calls invoke("check_brain_update") and returns its value', async () => {
    const payload = {
      available: true,
      commit: "abc123",
      brain_version: "0.0.44",
      reason: "available",
      detail: null,
    };
    mockInvoke.mockResolvedValueOnce(payload);

    const result = await checkBrainUpdate();

    expect(mockInvoke).toHaveBeenCalledWith("check_brain_update");
    expect(result).toEqual(payload);
  });
});

describe("applyBrainUpdate", () => {
  it('calls invoke("apply_brain_update") and returns its value', async () => {
    const payload = { commit: "abc123", brain_version: "0.0.44" };
    mockInvoke.mockResolvedValueOnce(payload);

    const result = await applyBrainUpdate();

    expect(mockInvoke).toHaveBeenCalledWith("apply_brain_update");
    expect(result).toEqual(payload);
  });
});

describe("rollbackBrain", () => {
  it('calls invoke("rollback_brain", { reason: "why" })', async () => {
    mockInvoke.mockResolvedValueOnce(undefined);

    await rollbackBrain("why");

    expect(mockInvoke).toHaveBeenCalledWith("rollback_brain", { reason: "why" });
  });
});

describe("revertBrain", () => {
  it('calls invoke("revert_brain")', async () => {
    mockInvoke.mockResolvedValueOnce(undefined);

    await revertBrain();

    expect(mockInvoke).toHaveBeenCalledWith("revert_brain");
  });
});

describe("ensureBridgeRunning", () => {
  it("drops the persona's cached bridge credentials, since a spawned bridge writes new ones (#365)", async () => {
    const { getBridgeCredentials, resetBridgeCredentialCache } = await import("./bridge");
    const { ensureBridgeRunning } = await import("./appConfig");
    resetBridgeCredentialCache();
    mockInvoke.mockImplementation(async (cmd) =>
      cmd === "get_bridge_credentials" ? { port: 41001, auth_token: "tok" } : undefined,
    );

    await getBridgeCredentials("alice");
    await ensureBridgeRunning("alice");
    await getBridgeCredentials("alice");

    const credentialReads = mockInvoke.mock.calls.filter(([cmd]) => cmd === "get_bridge_credentials");
    expect(credentialReads).toHaveLength(2);
  });
});

describe("ensureBridgeRunning: a start already under way (#456 review R2-M1)", () => {
  const EXIT_2 = "supervisor start failed (exit 2): bridge already starting (lockfile held)";

  function healthAfter(failures: number) {
    let n = 0;
    const fetchMock = vi.fn(async () => {
      n += 1;
      if (n <= failures) throw new TypeError("fetch failed");
      return new Response(JSON.stringify({ liveness: "ok" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  beforeEach(async () => {
    const { resetBridgeCredentialCache } = await import("./bridge");
    resetBridgeCredentialCache();
    vi.useFakeTimers();
    vi.stubEnv("DEV", false);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("exit 2 (the service's own start holds the lock) waits for /health instead of failing", async () => {
    const { ensureBridgeRunning } = await import("./appConfig");
    mockInvoke.mockImplementation(async (cmd) => {
      if (cmd === "get_bridge_credentials") return { port: 41001, auth_token: "tok" };
      if (cmd === "ensure_bridge_running") throw EXIT_2;
      return undefined;
    });
    const fetchMock = healthAfter(10); // five failed probes (each one is retried once with fresh credentials)
    const settled = vi.fn();
    const p = ensureBridgeRunning("alice");
    p.then(settled, settled);
    await vi.advanceTimersByTimeAsync(1000);
    expect(settled).not.toHaveBeenCalled(); // /health not answering yet: still waiting, not failed
    await vi.advanceTimersByTimeAsync(5000);
    await expect(p).resolves.toBeUndefined();
    expect(fetchMock.mock.calls.length).toBeGreaterThan(10);
  });

  it("exit 2 with a bridge that never answers fails after the wait, with the start's error", async () => {
    const { ensureBridgeRunning } = await import("./appConfig");
    mockInvoke.mockImplementation(async (cmd) => {
      if (cmd === "get_bridge_credentials") return { port: 41001, auth_token: "tok" };
      if (cmd === "ensure_bridge_running") throw EXIT_2;
      return undefined;
    });
    healthAfter(Number.MAX_SAFE_INTEGER);
    const p = ensureBridgeRunning("alice");
    const caught = p.catch((e) => e);
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS + 5000);
    expect(await caught).toBe(EXIT_2);
  });

  it("any other start failure is still a failure, with no wait", async () => {
    const { ensureBridgeRunning } = await import("./appConfig");
    mockInvoke.mockImplementation(async (cmd) => {
      if (cmd === "ensure_bridge_running") throw "supervisor start failed (exit 1): persona directory not found";
      return undefined;
    });
    const fetchMock = healthAfter(0);
    await expect(ensureBridgeRunning("alice")).rejects.toMatch(/exit 1/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
