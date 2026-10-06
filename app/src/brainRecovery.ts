/**
 * Recovering a brain update that didn't prove itself (#335).
 *
 * recoverUnhealthyBrain is the one rollback chain, shared by useBrainUpdate.apply
 * and the launch path: undo (rollback, or revert when the release brain was running
 * before the update), restart; still unhealthy → the release brain, restart.
 *
 * Spec: docs/source-spec/2026-10-01-brain-update-recovery-design.md
 */

import {
  brainOverlayStatus,
  confirmBrainUpdate,
  ensureBridgeRunning,
  revertBrain,
  rollbackBrain,
  type BrainOverlayStatus,
} from "./appConfig";
import { fetchHealth } from "./bridge";
import { ensureBridgeCurrent } from "./bridgeVersionCheck";
import { clearRestartState, restartBridge } from "./hooks/useRestartBridge";
import { errString } from "./lib/errString";

export type UndoKind = BrainOverlayStatus["undo"];

export interface RecoveryResult {
  /** the last restart came back healthy */
  ok: boolean;
  /** a failed switch to the release brain ("Couldn't switch back…") */
  error: string | null;
  /** where it ended: an earlier overlay, the release brain, or nothing healthy */
  outcome: "rolled_back" | "release" | "failed";
}

export async function recoverUnhealthyBrain(
  undo: UndoKind,
  reason: string,
  restart: () => Promise<boolean>,
): Promise<RecoveryResult> {
  let undoError: string | null = null;
  let landedOn: string | null = null; // the overlay a rollback landed on; null = release
  try {
    if (undo === "rollback") landedOn = await rollbackBrain(reason);
    else await revertBrain();
  } catch (e) {
    undoError = errString(e) || "unknown error";
  }
  // A failed undo leaves the still-broken build in place — restarting onto it
  // would just repeat the failure, so skip that restart.
  if (undoError === null && (await restart())) {
    return { ok: true, error: null, outcome: landedOn ? "rolled_back" : "release" };
  }
  // undo "revert": the revert WAS the release-brain step, so don't revert twice.
  let revertError: string | null = undo === "rollback" ? null : undoError;
  if (undo === "rollback") {
    try {
      await revertBrain();
    } catch (e) {
      revertError = errString(e) || "unknown error";
    }
  }
  // The release brain is the floor — always try the restart, even after a
  // failed revert, so a still-good overlay/release brain gets one more shot.
  const ok = await restart();
  // a failed revert left some overlay active, so it isn't the release brain running
  return { ok, error: revertError, outcome: !ok ? "failed" : revertError === null ? "release" : "rolled_back" };
}

export const NOTICE_ROLLED_BACK = "The brain update didn't start, so the previous brain is back.";
export const NOTICE_RELEASE =
  "The brain update didn't start, so Companion Emergence is on the release brain.";
export const STATUS_TIMEOUT_MS = 5000;

const NO_OVERLAY: BrainOverlayStatus = { active_commit: null, confirmed: true, undo: "revert" };

export type LaunchResult =
  | { kind: "ready"; versionMismatch: boolean; notice: string | null }
  | { kind: "error"; error: string; canUseReleaseBrain: boolean };

/** The quick launch check: null when the release-brain `nell` fails or doesn't answer
 *  within 5 s. A healthy launch then carries on as "no overlay" — it must not wait on
 *  it (spec §5); a failed start asks again with patientStatus(). */
async function quickStatus(): Promise<BrainOverlayStatus | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      brainOverlayStatus(),
      new Promise<null>((resolve) => {
        timer = setTimeout(() => resolve(null), STATUS_TIMEOUT_MS);
      }),
    ]);
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** After a failed start the user is already looking at an error, so waiting costs
 *  nothing: a slow machine must not switch recovery off (#338 review). Bounded by
 *  Rust's 30 s; a failure there is "no overlay", today's behaviour. */
async function patientStatus(): Promise<BrainOverlayStatus> {
  try {
    return await brainOverlayStatus(true);
  } catch {
    return NO_OVERLAY;
  }
}

/** What the running bridge loaded, and whether the user's escape hatch
 *  (KINDLED_NO_OVERLAY) has it ignoring overlays on purpose. */
async function runningOverlay(persona: string): Promise<{ commit: string | null; disabled: boolean }> {
  try {
    const health = await fetchHealth(persona);
    return { commit: health.overlay?.commit ?? null, disabled: health.overlay_disabled === true };
  } catch {
    return { commit: null, disabled: false };
  }
}

/** A restart the launch made, not the user: keep it off the Restart button. */
async function restartQuietly(persona: string): Promise<boolean> {
  try {
    return await restartBridge(persona);
  } finally {
    clearRestartState();
  }
}

function noticeFor(r: RecoveryResult): string {
  return r.outcome === "rolled_back" ? NOTICE_ROLLED_BACK : NOTICE_RELEASE;
}

async function confirmQuietly(commit: string): Promise<void> {
  try {
    await confirmBrainUpdate(commit);
  } catch (e) {
    // the build stays unconfirmed; the next healthy launch confirms it (spec §5)
    console.warn("[brainRecovery] confirm failed:", e);
  }
}

async function startOk(persona: string): Promise<boolean> {
  try {
    await ensureBridgeRunning(persona);
    return true;
  } catch {
    return false;
  }
}

async function failedRecovery(r: RecoveryResult, fallback: string): Promise<LaunchResult> {
  const after = await patientStatus();
  return {
    kind: "error",
    error: r.error !== null ? `Couldn't switch back to the release brain: ${r.error}` : fallback,
    canUseReleaseBrain: after.active_commit !== null,
  };
}

async function ready(persona: string, notice: string | null): Promise<LaunchResult> {
  const v = await ensureBridgeCurrent(persona);
  return { kind: "ready", versionMismatch: v === "version_mismatch_unresolved", notice };
}

/**
 * Start the persona's bridge, finishing or undoing an interrupted brain update
 * (spec §4). Unconfirmed builds are undone automatically; a confirmed one is
 * never dropped on a guess — the error screen offers the release brain instead.
 */
export async function launchBrain(persona: string): Promise<LaunchResult> {
  const quickP = quickStatus(); // runs alongside the start
  try {
    await ensureBridgeRunning(persona);
  } catch (e) {
    const error = errString(e) || "the bridge didn't start";
    const status = (await quickP) ?? (await patientStatus());
    if (!status.active_commit) return { kind: "error", error, canUseReleaseBrain: false };
    if (status.confirmed) return { kind: "error", error, canUseReleaseBrain: true };
    // case C: the update never proved itself and won't start
    const r = await recoverUnhealthyBrain(
      status.undo,
      "unconfirmed overlay failed to start at launch",
      () => startOk(persona),
    );
    if (!r.ok) return failedRecovery(r, error);
    return ready(persona, noticeFor(r));
  }

  const status = (await quickP) ?? NO_OVERLAY;
  let notice: string | null = null;
  if (status.active_commit) {
    const commit = status.active_commit;
    const running = await runningOverlay(persona);
    if (running.disabled) {
      // the user's escape hatch: leave the overlay alone, no restart, no confirm
    } else if (running.commit === commit) {
      // silent (spec §3.3): never hold the launch on it
      if (!status.confirmed) void confirmQuietly(commit);
    } else {
      // case A: the update installed but the bridge never restarted onto it
      const restarted = await restartQuietly(persona);
      if (restarted && (await runningOverlay(persona)).commit === commit) {
        void confirmQuietly(commit);
      } else if (status.confirmed) {
        if (!restarted) {
          return { kind: "error", error: "The brain didn't come back after a restart.", canUseReleaseBrain: true };
        }
        // healthy on another build: leave it running, never drop a proven build on a guess
      } else {
        const r = await recoverUnhealthyBrain(
          status.undo,
          "unfinished brain update didn't load at launch",
          () => restartQuietly(persona),
        );
        if (!r.ok) return failedRecovery(r, "The brain update didn't load.");
        notice = noticeFor(r);
      }
    }
  }
  return ready(persona, notice);
}
