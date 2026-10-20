// TDD: boot-time bridge version handshake — V2
// Start with one failing test; expand after each GREEN.

import { describe, it, expect, vi, beforeEach } from "vitest";

// ── Mocks ────────────────────────────────────────────────────────────────────

const { fetchHealth, shutdownBridge, invokeForceRestart } = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  shutdownBridge: vi.fn(async () => undefined),
  invokeForceRestart: vi.fn(async () => undefined),
}));

vi.mock("./bridge", () => ({
  fetchHealth,
  shutdownBridge,
  invokeForceRestart,
}));

// #456 (Q-A): the replace goes through the app's restart (stop, wait with the display and offer, then start).
const { restartBridge, clearRestartState } = vi.hoisted(() => ({
  restartBridge: vi.fn(async (_persona: string) => true),
  clearRestartState: vi.fn(),
}));

vi.mock("./hooks/useRestartBridge", () => ({
  restartBridge,
  clearRestartState,
}));

const { ensureBridgeRunning } = vi.hoisted(() => ({
  ensureBridgeRunning: vi.fn(async () => undefined),
}));

vi.mock("./appConfig", () => ({
  ensureBridgeRunning,
}));

const { getVersion } = vi.hoisted(() => ({
  getVersion: vi.fn(async () => "0.0.33"),
}));

vi.mock("@tauri-apps/api/app", () => ({
  getVersion,
}));

// ── Import under test (after mocks) ─────────────────────────────────────────

import { ensureBridgeCurrent, _resetForTests } from "./bridgeVersionCheck";

// ── Shared reset ─────────────────────────────────────────────────────────────

beforeEach(() => {
  _resetForTests();
  vi.clearAllMocks();
  getVersion.mockResolvedValue("0.0.33");
});

// ── Tests ────────────────────────────────────────────────────────────────────

describe("ensureBridgeCurrent", () => {
  it('version matches → "ok", no shutdown or ensure called', async () => {
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("ok");
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
    expect(restartBridge).not.toHaveBeenCalled();
  });

  it('version mismatch then match → "restarted", restartBridge called once, no shutdownBridge/ensureBridgeRunning', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.33" }) // stale
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.34" }); // after restart

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("restarted");
    expect(restartBridge).toHaveBeenCalledTimes(1);
    expect(restartBridge).toHaveBeenCalledWith("nell");
    expect(invokeForceRestart).not.toHaveBeenCalled(); // no kill of its own (#456 D10)
    expect(clearRestartState).toHaveBeenCalled(); // a launch-time restart nobody pressed (#335)
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
  });

  it('(#456) a restart that does not bring a bridge back → "version_mismatch_unresolved", no recheck', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });
    restartBridge.mockResolvedValueOnce(false);

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    expect(fetchHealth).toHaveBeenCalledTimes(1);
    expect(clearRestartState).toHaveBeenCalled();
  });

  it('persistent mismatch after restart → "version_mismatch_unresolved", one attempt only', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" }); // never matches

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    expect(restartBridge).toHaveBeenCalledTimes(1);
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
  });

  it("second call after mismatch_unresolved → no further restarts", async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    // First call — sets the guard flag
    await ensureBridgeCurrent("nell");
    vi.clearAllMocks();

    // Second call — flag already set, must not attempt again
    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    expect(restartBridge).not.toHaveBeenCalled();
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
  });

  it('getVersion throws → "skipped", no health check', async () => {
    getVersion.mockRejectedValue(new Error("unavailable"));

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(fetchHealth).not.toHaveBeenCalled();
  });

  it('fetchHealth rejects → "skipped", no shutdown', async () => {
    fetchHealth.mockRejectedValue(new Error("bridge not reachable"));

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(restartBridge).not.toHaveBeenCalled();
  });

  it('health.version undefined → treats as mismatch, calls restartBridge', async () => {
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok" })                     // no version field
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.33" }); // after restart

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("restarted");
    expect(restartBridge).toHaveBeenCalledTimes(1);
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
  });

  it("_resetForTests clears the flag so a fresh mismatch triggers a new attempt", async () => {
    // First attempt — mismatch + restart → still mismatched
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.32" });
    await ensureBridgeCurrent("nell");

    vi.clearAllMocks();
    _resetForTests();

    // After reset, health now resolves cleanly
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });
    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("ok");
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(restartBridge).not.toHaveBeenCalled();
  });

  // ── New test 1: dev-build skip ────────────────────────────────────────────

  it('appVersion "0.0.0" (dev build) → "skipped", no restart', async () => {
    getVersion.mockResolvedValue("0.0.0");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(restartBridge).not.toHaveBeenCalled();
  });

  // ── New test 2: non-semver appVersion → skipped ───────────────────────────

  it('appVersion non-semver ("xyz") → "skipped", no restart', async () => {
    getVersion.mockResolvedValue("xyz");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(restartBridge).not.toHaveBeenCalled();
  });

  // ── New test 3: garbage health.version → skipped ─────────────────────────

  it('health.version present but garbage ("weird") → "skipped", no restart', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "weird" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(restartBridge).not.toHaveBeenCalled();
  });

  // ── New test 4: 4-part version normalises equal → "ok" ───────────────────

  it('app "0.0.34", health "0.0.34.0" (4-part) → normalises equal → "ok", no restart', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.34.0" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("ok");
    expect(restartBridge).not.toHaveBeenCalled();
  });

  // ── New test 5: restartBridge throws → immediate unresolved ──────────

  it('restartBridge throws → "version_mismatch_unresolved" immediately, no recheck fetchHealth', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });
    restartBridge.mockRejectedValue(new Error("pid not found"));

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    // fetchHealth should not be called again — no recheck after failed restart
    expect(fetchHealth).toHaveBeenCalledTimes(1);
  });

  // ── New tests: overlay-aware version handshake (#286) ────────────────────

  it('app "0.0.43", overlay brain equal to overlay.brain_version → "ok", no restart', async () => {
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({
      liveness: "ok",
      version: "0.0.44",
      overlay: { commit: "a", brain_version: "0.0.44" },
    });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("ok");
    expect(restartBridge).not.toHaveBeenCalled();
  });

  it('app "0.0.43", overlay present but health.version OLDER than app → restart path', async () => {
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({
      liveness: "ok",
      version: "0.0.42",
      overlay: { commit: "a", brain_version: "0.0.42" },
    });

    await ensureBridgeCurrent("nell");

    expect(restartBridge).toHaveBeenCalledTimes(1);
  });

  it('app "0.0.43", NEWER health.version but no overlay → restart path (unchanged rule)', async () => {
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.44" });

    await ensureBridgeCurrent("nell");

    expect(restartBridge).toHaveBeenCalledTimes(1);
  });

  it('app "0.0.43", EQUAL version but overlay built for another bundle → restart path', async () => {
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({
      liveness: "ok",
      version: "0.0.43",
      overlay: { commit: "a", brain_version: "0.0.43", bundle_match: false },
    });

    await ensureBridgeCurrent("nell");

    expect(restartBridge).toHaveBeenCalledTimes(1);
  });

  it('app "0.0.43", NEWER overlay built for another bundle → restart path', async () => {
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({
      liveness: "ok",
      version: "0.0.44",
      overlay: { commit: "a", brain_version: "0.0.44", bundle_match: false },
    });

    await ensureBridgeCurrent("nell");

    expect(restartBridge).toHaveBeenCalledTimes(1);
  });

  it('app "0.0.43", NEWER overlay built for this bundle → "ok", no restart', async () => {
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({
      liveness: "ok",
      version: "0.0.44",
      overlay: { commit: "a", brain_version: "0.0.44", bundle_match: true },
    });

    expect(await ensureBridgeCurrent("nell")).toBe("ok");
    expect(restartBridge).not.toHaveBeenCalled();
  });

  it("(#365) says why when the forced replace of a stale bridge fails", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.30" });
    restartBridge.mockRejectedValueOnce(new Error("no pid file for nell"));

    expect(await ensureBridgeCurrent("nell")).toBe("version_mismatch_unresolved");

    expect(warn).toHaveBeenCalledWith(expect.stringContaining("forced replace failed"));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("no pid file for nell"));
  });

  it("(#365) says why when the recheck after a replace fails or still disagrees", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    restartBridge.mockResolvedValue(undefined); // an earlier test leaves a rejection in place
    getVersion.mockResolvedValue("0.0.43");
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.30" })
      .mockRejectedValueOnce(new Error("connection refused"));

    expect(await ensureBridgeCurrent("nell")).toBe("version_mismatch_unresolved");
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("recheck failed: connection refused"));

    _resetForTests();
    warn.mockClear();
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.30" })
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.31" });

    expect(await ensureBridgeCurrent("nell")).toBe("version_mismatch_unresolved");
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("recheck still reports 0.0.31"));
  });
});
