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
import { restartBridge } from "./hooks/useRestartBridge";
import { errString } from "./lib/errString";

export type UndoKind = BrainOverlayStatus["undo"];

export interface RecoveryResult {
  /** the last restart came back healthy */
  ok: boolean;
  /** a failed switch to the release brain ("Couldn't switch back…") */
  error: string | null;
}

export async function recoverUnhealthyBrain(
  undo: UndoKind,
  reason: string,
  restart: () => Promise<boolean>,
): Promise<RecoveryResult> {
  let undoError: string | null = null;
  try {
    if (undo === "rollback") await rollbackBrain(reason);
    else await revertBrain();
  } catch (e) {
    undoError = errString(e) || "unknown error";
  }
  // A failed undo leaves the still-broken build in place — restarting onto it
  // would just repeat the failure, so skip that restart.
  if (undoError === null && (await restart())) return { ok: true, error: null };
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
  return { ok: await restart(), error: revertError };
}

export const NOTICE_ROLLED_BACK = "The brain update didn't start, so the previous brain is back.";
export const NOTICE_RELEASE =
  "The brain update didn't start, so Companion Emergence is on the release brain.";
export const STATUS_TIMEOUT_MS = 5000;

const NO_OVERLAY: BrainOverlayStatus = { active_commit: null, confirmed: true, undo: "revert" };

export type LaunchResult =
  | { kind: "ready"; versionMismatch: boolean; notice: string | null }
  | { kind: "error"; error: string; canUseReleaseBrain: boolean };

/** The overlay status, or "no overlay" if the release-brain `nell` fails or is slow:
 *  that is today's behaviour, and the launch must not wait on it (spec §5). */
async function overlayStatus(): Promise<BrainOverlayStatus> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      brainOverlayStatus(),
      new Promise<BrainOverlayStatus>((resolve) => {
        timer = setTimeout(() => resolve(NO_OVERLAY), STATUS_TIMEOUT_MS);
      }),
    ]);
  } catch {
    return NO_OVERLAY;
  } finally {
    clearTimeout(timer);
  }
}

async function loadedCommit(persona: string): Promise<string | null> {
  try {
    return (await fetchHealth(persona)).overlay?.commit ?? null;
  } catch {
    return null;
  }
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

async function noticeAfterRecovery(): Promise<string> {
  return (await overlayStatus()).active_commit ? NOTICE_ROLLED_BACK : NOTICE_RELEASE;
}

async function failedRecovery(r: RecoveryResult, fallback: string): Promise<LaunchResult> {
  const after = await overlayStatus();
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
  const statusP = overlayStatus(); // runs alongside the start
  try {
    await ensureBridgeRunning(persona);
  } catch (e) {
    const error = errString(e) || "the bridge didn't start";
    const status = await statusP;
    if (!status.active_commit) return { kind: "error", error, canUseReleaseBrain: false };
    if (status.confirmed) return { kind: "error", error, canUseReleaseBrain: true };
    // case C: the update never proved itself and won't start
    const r = await recoverUnhealthyBrain(
      status.undo,
      "unconfirmed overlay failed to start at launch",
      () => startOk(persona),
    );
    if (!r.ok) return failedRecovery(r, error);
    return ready(persona, await noticeAfterRecovery());
  }

  const status = await statusP;
  let notice: string | null = null;
  if (status.active_commit) {
    const commit = status.active_commit;
    if ((await loadedCommit(persona)) === commit) {
      // silent (spec §3.3): never hold the launch on it
      if (!status.confirmed) void confirmQuietly(commit);
    } else {
      // case A: the update installed but the bridge never restarted onto it
      const restarted = await restartBridge(persona);
      if (restarted && (await loadedCommit(persona)) === commit) {
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
          () => restartBridge(persona),
        );
        if (!r.ok) return failedRecovery(r, "The brain update didn't load.");
        notice = await noticeAfterRecovery();
      }
    }
  }
  return ready(persona, notice);
}
