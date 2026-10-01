/**
 * Recovering a brain update that didn't prove itself (#335).
 *
 * recoverUnhealthyBrain is the one rollback chain, shared by useBrainUpdate.apply
 * and the launch path: undo (rollback, or revert when the release brain was running
 * before the update), restart; still unhealthy → the release brain, restart.
 *
 * Spec: docs/source-spec/2026-10-01-brain-update-recovery-design.md
 */

import { revertBrain, rollbackBrain, type BrainOverlayStatus } from "./appConfig";
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
