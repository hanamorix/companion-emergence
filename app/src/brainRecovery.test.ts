// #335 launch table. Spec: docs/source-spec/2026-10-01-brain-update-recovery-design.md §4.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import * as appConfig from "./appConfig";
import * as bridge from "./bridge";
import * as restartModule from "./hooks/useRestartBridge";
import {
  NOTICE_RELEASE,
  NOTICE_ROLLED_BACK,
  STATUS_TIMEOUT_MS,
  launchBrain,
} from "./brainRecovery";

vi.mock("./appConfig", () => ({
  brainOverlayStatus: vi.fn(),
  confirmBrainUpdate: vi.fn(async () => undefined),
  ensureBridgeRunning: vi.fn(async () => undefined),
  revertBrain: vi.fn(async () => undefined),
  rollbackBrain: vi.fn(async () => undefined),
}));
vi.mock("./bridge", () => ({ fetchHealth: vi.fn() }));
vi.mock("./bridgeVersionCheck", () => ({ ensureBridgeCurrent: vi.fn(async () => "ok") }));
vi.mock("./hooks/useRestartBridge", () => ({ restartBridge: vi.fn(async () => true) }));

const P = "nell";
const NEW = "b".repeat(40);
const OLD = "a".repeat(40);
const NONE: appConfig.BrainOverlayStatus = { active_commit: null, confirmed: true, undo: "revert" };

function status(...seq: appConfig.BrainOverlayStatus[]) {
  const m = vi.mocked(appConfig.brainOverlayStatus);
  for (const s of seq) m.mockResolvedValueOnce(s);
  m.mockResolvedValue(seq[seq.length - 1]);
}
function loaded(...commits: (string | null)[]) {
  const m = vi.mocked(bridge.fetchHealth);
  for (const c of commits) {
    m.mockResolvedValueOnce({ liveness: "ok", overlay: c ? { commit: c, brain_version: "0.0.44", bundle_match: true } : null });
  }
}
function starts(...results: boolean[]) {
  const m = vi.mocked(appConfig.ensureBridgeRunning);
  for (const ok of results) {
    if (ok) m.mockResolvedValueOnce(undefined);
    else m.mockRejectedValueOnce(new Error("supervisor_start_timeout"));
  }
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(appConfig.ensureBridgeRunning).mockReset().mockResolvedValue(undefined);
  vi.mocked(appConfig.brainOverlayStatus).mockReset();
  vi.mocked(bridge.fetchHealth).mockReset();
  vi.mocked(restartModule.restartBridge).mockReset().mockResolvedValue(true);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("launchBrain (#335)", () => {
  it("no active overlay: unchanged launch", async () => {
    status(NONE);
    expect(await launchBrain(P)).toEqual({ kind: "ready", versionMismatch: false, notice: null });
    expect(appConfig.confirmBrainUpdate).not.toHaveBeenCalled();
    expect(restartModule.restartBridge).not.toHaveBeenCalled();
  });

  it("no active overlay and the start fails: today's error screen, no release-brain button", async () => {
    status(NONE);
    starts(false);
    expect(await launchBrain(P)).toEqual({ kind: "error", error: "supervisor_start_timeout", canUseReleaseBrain: false });
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
  });

  it("healthy and loaded == active but unconfirmed → confirmed (also the state a launch interrupted mid-recovery leaves)", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" });
    loaded(NEW);
    expect((await launchBrain(P)).kind).toBe("ready");
    expect(appConfig.confirmBrainUpdate).toHaveBeenCalledWith(NEW);
  });

  it("healthy and loaded == active, already confirmed → nothing to do", async () => {
    status({ active_commit: NEW, confirmed: true, undo: "rollback" });
    loaded(NEW);
    await launchBrain(P);
    expect(appConfig.confirmBrainUpdate).not.toHaveBeenCalled();
  });

  it("case A: the old bridge still runs → restart onto the update, then confirm", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" });
    loaded(OLD, NEW);
    expect(await launchBrain(P)).toEqual({ kind: "ready", versionMismatch: false, notice: null });
    expect(restartModule.restartBridge).toHaveBeenCalledWith(P);
    expect(appConfig.confirmBrainUpdate).toHaveBeenCalledWith(NEW);
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
  });

  it("case A: the restart fails on an unconfirmed update → rolled back, with a notice", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" }, { active_commit: OLD, confirmed: true, undo: "revert" });
    loaded(OLD);
    vi.mocked(restartModule.restartBridge).mockResolvedValueOnce(false).mockResolvedValueOnce(true);
    expect(await launchBrain(P)).toEqual({ kind: "ready", versionMismatch: false, notice: NOTICE_ROLLED_BACK });
    expect(appConfig.rollbackBrain).toHaveBeenCalledWith("unfinished brain update didn't load at launch");
    expect(appConfig.confirmBrainUpdate).not.toHaveBeenCalled();
  });

  it("case A: a confirmed build whose restart fails is never dropped → error screen with the button", async () => {
    status({ active_commit: NEW, confirmed: true, undo: "rollback" });
    loaded(OLD);
    vi.mocked(restartModule.restartBridge).mockResolvedValueOnce(false);
    const r = await launchBrain(P);
    expect(r).toMatchObject({ kind: "error", canUseReleaseBrain: true });
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
    expect(appConfig.revertBrain).not.toHaveBeenCalled();
  });

  it("case C: an unconfirmed build that won't start → rolled back, retried, notice", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" }, { active_commit: OLD, confirmed: true, undo: "revert" });
    starts(false, true);
    expect(await launchBrain(P)).toEqual({ kind: "ready", versionMismatch: false, notice: NOTICE_ROLLED_BACK });
    expect(appConfig.rollbackBrain).toHaveBeenCalledWith("unconfirmed overlay failed to start at launch");
    expect(appConfig.ensureBridgeRunning).toHaveBeenCalledTimes(2);
  });

  it("case C: the rolled-back build fails too → the release brain, release notice", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" }, NONE);
    starts(false, false, true);
    expect(await launchBrain(P)).toEqual({ kind: "ready", versionMismatch: false, notice: NOTICE_RELEASE });
    expect(appConfig.rollbackBrain).toHaveBeenCalledTimes(1);
    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
  });

  it("case C with undo revert never rolls back (an older reverted-away overlay stays away)", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "revert" }, NONE);
    starts(false, true);
    expect(await launchBrain(P)).toEqual({ kind: "ready", versionMismatch: false, notice: NOTICE_RELEASE });
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
    expect(appConfig.revertBrain).toHaveBeenCalledTimes(1);
  });

  it("case C on a confirmed build: ask — error screen with the release-brain button", async () => {
    status({ active_commit: NEW, confirmed: true, undo: "rollback" });
    starts(false);
    expect(await launchBrain(P)).toEqual({ kind: "error", error: "supervisor_start_timeout", canUseReleaseBrain: true });
    expect(appConfig.rollbackBrain).not.toHaveBeenCalled();
  });

  it("recovery that can't bring anything up ends on the error screen", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" }, NONE);
    vi.mocked(appConfig.ensureBridgeRunning).mockRejectedValue(new Error("boom"));
    expect(await launchBrain(P)).toEqual({ kind: "error", error: "boom", canUseReleaseBrain: false });
  });

  it("a hung status check doesn't hold the launch", async () => {
    vi.useFakeTimers();
    vi.mocked(appConfig.brainOverlayStatus).mockReturnValue(new Promise(() => undefined));
    const pending = launchBrain(P);
    await vi.advanceTimersByTimeAsync(STATUS_TIMEOUT_MS + 10);
    expect(await pending).toEqual({ kind: "ready", versionMismatch: false, notice: null });
  });

  it("a failing status check counts as no overlay", async () => {
    vi.mocked(appConfig.brainOverlayStatus).mockRejectedValue(new Error("nell exploded"));
    expect((await launchBrain(P)).kind).toBe("ready");
    expect(bridge.fetchHealth).not.toHaveBeenCalled();
  });

  it("a slow confirm doesn't hold the launch (it is silent, spec §3.3)", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" });
    loaded(NEW);
    vi.mocked(appConfig.confirmBrainUpdate).mockReturnValueOnce(new Promise(() => undefined));
    const r = await Promise.race([
      launchBrain(P),
      new Promise((resolve) => setTimeout(() => resolve("held"), 200)),
    ]);
    expect(r).toMatchObject({ kind: "ready" });
    expect(appConfig.confirmBrainUpdate).toHaveBeenCalledWith(NEW);
  });

  it("a failed confirm still launches (the next healthy launch confirms)", async () => {
    status({ active_commit: NEW, confirmed: false, undo: "rollback" });
    loaded(NEW);
    vi.mocked(appConfig.confirmBrainUpdate).mockRejectedValueOnce(new Error("locked"));
    expect((await launchBrain(P)).kind).toBe("ready");
  });
});
