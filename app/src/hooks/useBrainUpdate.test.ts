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
import { _resetBrainUpdateForTests, useBrainUpdate } from "./useBrainUpdate";
import type { BrainUpdateCheck } from "../appConfig";
import type { BridgeHealth } from "../bridge";

vi.mock("../appConfig", () => ({
  checkBrainUpdate: vi.fn(),
  applyBrainUpdate: vi.fn(),
  rollbackBrain: vi.fn(),
  revertBrain: vi.fn(),
  confirmBrainUpdate: vi.fn(async () => undefined),
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
  _resetBrainUpdateForTests();
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

  it("check() with reason bad_signature says it couldn't be verified, not the crypto detail", async () => {
    // A publish in progress can briefly pair a new manifest with an old .sig; the
    // raw verifier message goes to the security log, the user gets a calm retry hint.
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: false, reason: "bad_signature", detail: "sig mismatch" }),
    );

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    expect(result.current.state).toEqual({
      kind: "error",
      detail:
        "The brain update couldn't be verified. If a new build is being published right now, check again in a minute.",
    });
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
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    // the restarted bridge runs the new build
    vi.mocked(bridge.fetchHealth).mockResolvedValue(health({ commit, brain_version: "0.0.44" }));
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.applyBrainUpdate).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(1);
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
    expect(bridge.fetchHealth).toHaveBeenCalledWith(PERSONA);
    expect(result.current.state).toEqual({ kind: "none", reason: "already_active" });
    expect(appConfig.confirmBrainUpdate).toHaveBeenCalledWith(commit);
  });

  it("apply() reports an error when the restarted brain isn't the update", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: false });
    mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    vi.mocked(bridge.fetchHealth).mockResolvedValue(health({ commit: "d".repeat(40), brain_version: "0.0.43" }));
    await act(async () => {
      await result.current.apply();
    });

    expect(result.current.state).toEqual({
      kind: "error",
      detail: "The update installed, but the brain didn't load it. Use the release brain below, or restart Companion Emergence.",
    });
    expect(appConfig.confirmBrainUpdate).not.toHaveBeenCalled();
  });

  it("apply() rolls back and restarts once when the first restart is unhealthy", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
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

  it("apply() with no earlier overlay undoes to the release brain, not a rollback", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: false });
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    const restart = mockRestart(false, true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(2);
    expect(result.current.state).toEqual({ kind: "rolled_back" });
  });

  it("apply() falls through to the release brain when the rollback restart is also unhealthy", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
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
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
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

  it("apply() called twice without awaiting the first is guarded — single applyBrainUpdate/restart call", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });

    await act(async () => {
      const p1 = result.current.apply();
      const p2 = result.current.apply();
      await Promise.all([p1, p2]);
    });

    expect(appConfig.applyBrainUpdate).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(1);
  });

  it("check() while an apply() is in flight is ignored", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
    mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    vi.mocked(appConfig.checkBrainUpdate).mockClear();

    await act(async () => {
      const p1 = result.current.apply();
      const p2 = result.current.check();
      await Promise.all([p1, p2]);
    });

    expect(appConfig.checkBrainUpdate).not.toHaveBeenCalled();
  });

  it("rollbackBrain rejecting skips the post-rollback restart and falls straight to the release brain", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
    vi.mocked(appConfig.rollbackBrain).mockRejectedValue(new Error("rollback failed"));
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    const restart = mockRestart(false, true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(appConfig.rollbackBrain).toHaveBeenCalledTimes(1);
    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(2);
    expect(result.current.state).toEqual({ kind: "rolled_back" });
  });

  it("revertBrain rejecting after a failed rollback-restart ends in an error naming the release brain", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    vi.mocked(appConfig.applyBrainUpdate).mockResolvedValue({ commit, brain_version: "0.0.44", had_active: true });
    vi.mocked(appConfig.rollbackBrain).mockResolvedValue(undefined);
    vi.mocked(appConfig.revertBrain).mockRejectedValue(new Error("revert failed"));
    const restart = mockRestart(false, false, true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await result.current.check();
    });
    await act(async () => {
      await result.current.apply();
    });

    expect(restart).toHaveBeenCalledTimes(3);
    expect(result.current.state.kind).toBe("error");
    expect((result.current.state as { detail: string }).detail).toMatch(
      /^Couldn't switch back to the release brain/,
    );
  });

  it("useReleaseBrain() called twice without awaiting the first is guarded — single revertBrain call", async () => {
    vi.mocked(appConfig.revertBrain).mockResolvedValue(undefined);
    const restart = mockRestart(true);

    const { result } = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      const p1 = result.current.useReleaseBrain();
      const p2 = result.current.useReleaseBrain();
      await Promise.all([p1, p2]);
    });

    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
    expect(restart).toHaveBeenCalledTimes(1);
  });

  it("a remount mid-apply shows the in-flight state and keeps actions no-ops", async () => {
    const commit = "c".repeat(40);
    vi.mocked(appConfig.checkBrainUpdate).mockResolvedValue(
      checkResult({ available: true, commit, brain_version: "0.0.44", reason: "available" }),
    );
    let finishApply: (v: appConfig.BrainUpdateApplied) => void = () => {};
    vi.mocked(appConfig.applyBrainUpdate).mockReturnValue(
      new Promise((resolve) => {
        finishApply = resolve;
      }),
    );
    mockRestart(true);

    const first = renderHook(() => useBrainUpdate(PERSONA, "live"));
    await act(async () => {
      await first.result.current.check();
    });
    let applying: Promise<void> = Promise.resolve();
    await act(async () => {
      applying = first.result.current.apply();
    });
    first.unmount();

    const second = renderHook(() => useBrainUpdate(PERSONA, "live"));
    expect(second.result.current.state).toEqual({ kind: "applying", commit });
    vi.mocked(appConfig.checkBrainUpdate).mockClear();
    await act(async () => {
      await second.result.current.check();
    });
    expect(appConfig.checkBrainUpdate).not.toHaveBeenCalled();
    expect(second.result.current.state).toEqual({ kind: "applying", commit });

    await act(async () => {
      finishApply({ commit, brain_version: "0.0.44", had_active: false });
      await applying;
    });
  });
});
