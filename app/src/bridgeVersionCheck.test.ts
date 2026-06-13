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
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  it('version mismatch then match → "restarted", invokeForceRestart called once, no shutdownBridge/ensureBridgeRunning', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.33" }) // stale
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.34" }); // after force-restart

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("restarted");
    expect(invokeForceRestart).toHaveBeenCalledTimes(1);
    expect(invokeForceRestart).toHaveBeenCalledWith("nell");
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
  });

  it('persistent mismatch after restart → "version_mismatch_unresolved", one attempt only', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" }); // never matches

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    expect(invokeForceRestart).toHaveBeenCalledTimes(1);
    expect(shutdownBridge).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).not.toHaveBeenCalled();
  });

  it("second call after mismatch_unresolved → no further force-restarts", async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    // First call — sets the guard flag
    await ensureBridgeCurrent("nell");
    vi.clearAllMocks();

    // Second call — flag already set, must not attempt again
    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    expect(invokeForceRestart).not.toHaveBeenCalled();
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
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  it('health.version undefined → treats as mismatch, calls invokeForceRestart', async () => {
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok" })                     // no version field
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.33" }); // after restart

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("restarted");
    expect(invokeForceRestart).toHaveBeenCalledTimes(1);
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
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  // ── New test 1: dev-build skip ────────────────────────────────────────────

  it('appVersion "0.0.0" (dev build) → "skipped", no force-restart', async () => {
    getVersion.mockResolvedValue("0.0.0");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  // ── New test 2: non-semver appVersion → skipped ───────────────────────────

  it('appVersion non-semver ("xyz") → "skipped", no force-restart', async () => {
    getVersion.mockResolvedValue("xyz");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  // ── New test 3: garbage health.version → skipped ─────────────────────────

  it('health.version present but garbage ("weird") → "skipped", no force-restart', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "weird" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("skipped");
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  // ── New test 4: 4-part version normalises equal → "ok" ───────────────────

  it('app "0.0.34", health "0.0.34.0" (4-part) → normalises equal → "ok", no restart', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.34.0" });

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("ok");
    expect(invokeForceRestart).not.toHaveBeenCalled();
  });

  // ── New test 5: invokeForceRestart throws → immediate unresolved ──────────

  it('invokeForceRestart throws → "version_mismatch_unresolved" immediately, no recheck fetchHealth', async () => {
    getVersion.mockResolvedValue("0.0.34");
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.33" });
    invokeForceRestart.mockRejectedValue(new Error("pid not found"));

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    // fetchHealth should not be called again — no recheck after failed force-restart
    expect(fetchHealth).toHaveBeenCalledTimes(1);
  });
});
