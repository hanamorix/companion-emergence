// TDD: boot-time bridge version handshake — V2
// Start with one failing test; expand after each GREEN.

import { describe, it, expect, vi, beforeEach } from "vitest";

// ── Mocks ────────────────────────────────────────────────────────────────────

const { fetchHealth, shutdownBridge } = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  shutdownBridge: vi.fn(async () => undefined),
}));

vi.mock("./bridge", () => ({
  fetchHealth,
  shutdownBridge,
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
  });

  it('version mismatch then match → "restarted", exactly one shutdown + one ensure', async () => {
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.32" }) // stale
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.33" }); // after restart

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("restarted");
    expect(shutdownBridge).toHaveBeenCalledTimes(1);
    expect(shutdownBridge).toHaveBeenCalledWith("nell");
    expect(ensureBridgeRunning).toHaveBeenCalledTimes(1);
    expect(ensureBridgeRunning).toHaveBeenCalledWith("nell");
  });

  it('persistent mismatch after restart → "version_mismatch_unresolved", one attempt only', async () => {
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.32" }); // never matches

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
    expect(shutdownBridge).toHaveBeenCalledTimes(1);
    expect(ensureBridgeRunning).toHaveBeenCalledTimes(1);
  });

  it("second call after mismatch_unresolved → no further shutdowns", async () => {
    fetchHealth.mockResolvedValue({ liveness: "ok", version: "0.0.32" });

    // First call — sets the guard flag
    await ensureBridgeCurrent("nell");
    vi.clearAllMocks();

    // Second call — flag already set, must not attempt again
    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("version_mismatch_unresolved");
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
  });

  it('health.version undefined → treats as mismatch, attempts restart', async () => {
    fetchHealth
      .mockResolvedValueOnce({ liveness: "ok" })                     // no version field
      .mockResolvedValueOnce({ liveness: "ok", version: "0.0.33" }); // after restart

    const result = await ensureBridgeCurrent("nell");

    expect(result).toBe("restarted");
    expect(shutdownBridge).toHaveBeenCalledTimes(1);
    expect(ensureBridgeRunning).toHaveBeenCalledTimes(1);
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
  });
});
