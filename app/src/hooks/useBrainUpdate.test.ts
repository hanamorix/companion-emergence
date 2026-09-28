// Vitest coverage for the #286 slice 4 brain-update hook. Mocks appConfig's
// wrappers, bridge's fetchHealth, and useRestartBridge wholesale so no real
// network call, restart flow, or timer runs.
//
// Spec: docs/superpowers/specs/2026-09-28-brain-update-app §4 + §6.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor, cleanup } from "@testing-library/react";

import * as appConfig from "../appConfig";
import * as bridge from "../bridge";
import * as useRestartBridgeModule from "./useRestartBridge";
import { useBrainUpdate } from "./useBrainUpdate";
import type { BrainUpdateCheck } from "../appConfig";
import type { BridgeHealth } from "../bridge";

vi.mock("../appConfig", () => ({
  checkBrainUpdate: vi.fn(),
  applyBrainUpdate: vi.fn(),
  rollbackBrain: vi.fn(),
  revertBrain: vi.fn(),
}));

vi.mock("../bridge", () => ({
  fetchHealth: vi.fn(),
}));

vi.mock("./useRestartBridge", () => ({
  useRestartBridge: vi.fn(),
}));

const PERSONA = "test-persona";

function health(overlay: BridgeHealth["overlay"] = null): BridgeHealth {
  return { liveness: "ok", overlay };
}

function checkResult(overrides: Partial<BrainUpdateCheck>): BrainUpdateCheck {
  return {
    available: false,
    commit: null,
    brain_version: null,
    reason: "already_active",
    detail: null,
    ...overrides,
  };
}

/** restart() resolves the given sequence of booleans, one per call. */
function mockRestart(...results: boolean[]): ReturnType<typeof vi.fn> {
  const restart = vi.fn();
  for (const r of results) restart.mockResolvedValueOnce(r);
  vi.mocked(useRestartBridgeModule.useRestartBridge).mockReturnValue({
    state: "idle",
    errorDetail: null,
    restart,
    onModeChanged: vi.fn(),
  });
  return restart;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(bridge.fetchHealth).mockResolvedValue(health());
  mockRestart(true);
});

afterEach(() => {
  vi.restoreAllMocks();
  cleanup();
});

describe("useBrainUpdate", () => {
  it("mount fetches health and stores the overlay", async () => {
    const overlay = { commit: "a", brain_version: "0.0.43" };
    vi.mocked(bridge.fetchHealth).mockResolvedValue(health(overlay));

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));

    await waitFor(() => expect(bridge.fetchHealth).toHaveBeenCalledWith(PERSONA));
    await waitFor(() => expect(result.current.overlay).toEqual(overlay));
  });

  it("check() with an available update sets state to available", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    expect(result.current.state).toEqual({
      kind: "available",
      commit,
      brainVersion: "0.0.44",
    });
  });

  it("check() with reason already_active sets state to none", async () => {
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: false, reason: "already_active" }),
    );

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    expect(result.current.state).toEqual({ kind: "none", reason: "already_active" });
  });

  it("check() with reason unreachable sets an error state with the detail folded in", async () => {
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: false, reason: "unreachable", detail: "x" }),
    );

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    expect(result.current.state).toEqual({
      kind: "error",
      detail: "Couldn't check for a brain update: x",
    });
  });

  it("check() with reason bad_signature sets an error state", async () => {
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: false, reason: "bad_signature", detail: "sig mismatch" }),
    );

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    expect(result.current.state.kind).toBe("error");
  });

  it("check() when checkBrainUpdate rejects sets an error state", async () => {
    vi.mocked(appConfig.checkBrainUpdate).mockRejectedValue(new Error("network down"));

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    expect(result.current.state.kind).toBe("error");
  });

  it("apply() succeeds when the restart comes back healthy", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44" });
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.applyBrainUpdate).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(1);
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
    expect(bridge.fetchHealth).toHaveBeenCalledWith(PERSONA);
    expect(result.current.state).toEqual({ kind: "none", reason: "already_active" });
  });

  it("apply() rolls back and restarts once when the first restart is unhealthy", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44" });
    vi.mocked(appConfig.rollbackBrain).mockResolvedValue(undefined);
    const restart = mockRestart(false, true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.rollbackBrain).toHaveBeenCalledWith(expect.any(String));
    expect(restart).toHaveBeenCalledTimes(2);
    expect(appConfig.revertBrain).not.toHaveBeenCalled();
    expect(result.current.state).toEqual({ kind: "rolled_back" });
  });

  it("apply() falls through to the release brain when the rollback restart is also unhealthy", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44" });
    vi.mocked(appConfig.rollbackBrain).mockResolvedValue(undefined);
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    const restart = mockRestart(false, false, true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.rollbackBrain).toHaveBeenCalledTimes(1);
    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(3);
    expect(result.current.state).toEqual({ kind: "rolled_back" });
  });

  it("apply() ends in error when every restart in the chain fails", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44" });
    vi.mocked(appConfig.rollbackBrain).mockResolvedValue(undefined);
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    mockRestart(false, false, false);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(result.current.state.kind).toBe("error");
    expect((result.current.state as { detail: string }).detail).toMatch(/restart/i);
  });

  it("apply() surfaces an error and never restarts when applyBrainUpdate rejects", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockRejectedValue(new Error("boom"));
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(result.current.state).toEqual({
      kind: "error",
      detail: "Couldn't update the brain: boom",
    });
    expect(restart).not.toHaveBeenCalled();
  });

  it("apply() is a no-op when state isn't available", async () => {
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.applyBrainUpdate).not.toHaveBeenCalled();
    expect(restart).not.toHaveBeenCalled();
  });

  it("useReleaseBrain() reverts and restarts, ending idle on a healthy restart", async () => {
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.useReleaseBrain();
    });

    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(1);
    expect(bridge.fetchHealth).toHaveBeenCalledWith(PERSONA);
    expect(result.current.state).toEqual({ kind: "idle" });
  });

  it("useReleaseBrain() ends in error when the restart is unhealthy", async () => {
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    mockRestart(false);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.useReleaseBrain();
    });

    expect(result.current.state.kind).toBe("error");
  });
});
