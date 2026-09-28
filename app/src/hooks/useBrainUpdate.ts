/**
 * useBrainUpdate — brain updates from main (#286 slice 4, spec §4 + §6).
 *
 * check → (Update) apply: Rust downloads + verifies + `nell update` installs into
 * the overlay → the Restart button's graceful flow restarts the bridge onto it.
 * Unhealthy → roll back and restart; still unhealthy → the release brain and
 * restart. The release brain is the floor.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import {
  applyBrainUpdate,
  checkBrainUpdate,
  revertBrain,
  rollbackBrain,
  type BrainUpdateApplied,
} from "../appConfig";
import { fetchHealth } from "../bridge";
import type { BridgeOverlay, PersonaState } from "../bridge";
import { errString } from "../lib/errString";
import { useRestartBridge } from "./useRestartBridge";

export type BrainUpdateState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "none"; reason: string }
  | { kind: "available"; commit: string; brainVersion: string }
  | { kind: "applying"; commit: string }
  | { kind: "restarting" }
  | { kind: "reverting" }
  | { kind: "rolled_back" }
  | { kind: "error"; detail: string };

export interface UseBrainUpdate {
  state: BrainUpdateState;
  /** What the running bridge uses (from /health.overlay); null = the release brain. */
  overlay: BridgeOverlay | null;
  check: () => Promise<void>;
  apply: () => Promise<void>;
  useReleaseBrain: () => Promise<void>;
}

const NOT_LOADED =
  "The update installed, but the brain didn't load it. Use the release brain below, or restart Companion Emergence.";

const RESTART_FAILED =
  "The brain didn't come back after the update. Try Restart, or restart Companion Emergence.";

// Module scope, not per-instance: an update outlives the panel that started it,
// so a remount mid-update shows the in-flight state and `busy` keeps its
// actions no-ops. `busy` guards check/apply/useReleaseBrain against re-entry
// from a double-click before the next render lands, and stops check() from
// clobbering state mid-apply.
// ponytail: one app-wide update at a time — the overlay is app-wide too.
let shared: { state: BrainUpdateState; busy: boolean } = { state: { kind: "idle" }, busy: false };
const listeners = new Set<() => void>();

function setShared(state: BrainUpdateState): void {
  shared = { ...shared, state };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Reset for test isolation — never call in production code. */
export function _resetBrainUpdateForTests(): void {
  shared = { state: { kind: "idle" }, busy: false };
}

export function useBrainUpdate(
  persona: string,
  mode: PersonaState["mode"],
): UseBrainUpdate {
  const state = useSyncExternalStore(subscribe, () => shared.state);
  const [overlay, setOverlay] = useState<BridgeOverlay | null>(null);
  const overlayRef = useRef<BridgeOverlay | null>(null);
  const { restart } = useRestartBridge(persona, mode);

  /** Re-read /health.overlay; returns what it read (the previous one on error). */
  const refreshOverlay = useCallback(async (): Promise<BridgeOverlay | null> => {
    try {
      const health = await fetchHealth(persona);
      overlayRef.current = health.overlay ?? null;
      setOverlay(overlayRef.current);
    } catch {
      // Bridge not reachable right now — keep what we had.
    }
    return overlayRef.current;
  }, [persona]);

  useEffect(() => {
    void refreshOverlay();
  }, [refreshOverlay]);

  const check = useCallback(async () => {
    if (shared.busy) return;
    shared.busy = true;
    try {
      setShared({ kind: "checking" });
      try {
        const res = await checkBrainUpdate();
        await refreshOverlay();
        if (res.available && res.commit && res.brain_version) {
          setShared({ kind: "available", commit: res.commit, brainVersion: res.brain_version });
        } else if (res.reason === "unreachable" || res.reason === "bad_signature") {
          setShared({ kind: "error", detail: `Couldn't check for a brain update: ${res.detail || res.reason}` });
        } else {
          setShared({ kind: "none", reason: res.reason });
        }
      } catch (e) {
        setShared({ kind: "error", detail: `Couldn't check for a brain update: ${errString(e) || "unknown error"}` });
      }
    } finally {
      shared.busy = false;
    }
  }, [refreshOverlay]);

  const apply = useCallback(async () => {
    const current = shared.state;
    if (current.kind !== "available") return;
    if (shared.busy) return;
    shared.busy = true;
    try {
      setShared({ kind: "applying", commit: current.commit });
      let applied: BrainUpdateApplied;
      try {
        applied = await applyBrainUpdate();
      } catch (e) {
        setShared({ kind: "error", detail: `Couldn't update the brain: ${errString(e) || "unknown error"}` });
        return;
      }
      setShared({ kind: "restarting" });
      if (await restart()) {
        // healthy isn't enough: the running bridge must be on the new build
        const running = await refreshOverlay();
        setShared(running?.commit !== applied.commit
          ? { kind: "error", detail: NOT_LOADED }
          : { kind: "none", reason: "already_active" });
        return;
      }
      // §6: the updated bridge is unhealthy → undo and restart; if that is
      // unhealthy too, the release brain and restart. Undo = roll back to the
      // overlay that was active before, or — when there was none — the release
      // brain itself (a rollback there would keep this broken build as the
      // rollback target).
      let undoError: string | null = null;
      try {
        if (applied.had_active) await rollbackBrain("bridge unhealthy after a brain update");
        else await revertBrain();
      } catch (e) {
        undoError = errString(e) || "unknown error";
      }
      // A failed undo leaves the still-broken build in place — restarting onto
      // it would just repeat the failure, so skip that restart.
      if (undoError === null && (await restart())) {
        await refreshOverlay();
        setShared({ kind: "rolled_back" });
        return;
      }
      // had_active false: the revert WAS the undo, so don't revert twice.
      let revertError: string | null = applied.had_active ? null : undoError;
      if (applied.had_active) {
        try {
          await revertBrain();
        } catch (e) {
          revertError = errString(e) || "unknown error";
        }
      }
      // The release brain is the floor — always try the restart, even after a
      // failed revert, so a still-good overlay/release brain gets one more shot.
      const ok = await restart();
      await refreshOverlay();
      setShared(revertError !== null
        ? {
            kind: "error",
            detail: `Couldn't switch back to the release brain: ${revertError}. Try Restart, or restart Companion Emergence.`,
          }
        : ok ? { kind: "rolled_back" } : { kind: "error", detail: RESTART_FAILED });
    } finally {
      shared.busy = false;
    }
  }, [restart, refreshOverlay]);

  const useReleaseBrain = useCallback(async () => {
    if (shared.busy) return;
    shared.busy = true;
    try {
      setShared({ kind: "reverting" });
      try {
        await revertBrain();
      } catch (e) {
        setShared({ kind: "error", detail: `Couldn't switch to the release brain: ${errString(e) || "unknown error"}` });
        return;
      }
      const ok = await restart();
      await refreshOverlay();
      setShared(ok ? { kind: "idle" } : { kind: "error", detail: RESTART_FAILED });
    } finally {
      shared.busy = false;
    }
  }, [restart, refreshOverlay]);

  return { state, overlay, check, apply, useReleaseBrain };
}
