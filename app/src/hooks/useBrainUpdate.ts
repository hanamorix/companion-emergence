/**
 * useBrainUpdate — brain updates from main (#286 slice 4, spec §4 + §6).
 *
 * check → (Update) apply: Rust downloads + verifies + `nell update` installs into
 * the overlay → the Restart button's graceful flow restarts the bridge onto it.
 * Unhealthy → roll back and restart; still unhealthy → the release brain and
 * restart. The release brain is the floor.
 */

import { useCallback, useEffect, useState } from "react";
import {
  applyBrainUpdate,
  checkBrainUpdate,
  revertBrain,
  rollbackBrain,
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

const RESTART_FAILED =
  "The brain didn't come back after the update. Try Restart, or restart Companion Emergence.";

export function useBrainUpdate(
  persona: string,
  mode: PersonaState["mode"],
): UseBrainUpdate {
  const [state, setState] = useState<BrainUpdateState>({ kind: "idle" });
  const [overlay, setOverlay] = useState<BridgeOverlay | null>(null);
  const { restart } = useRestartBridge(persona, mode);

  const refreshOverlay = useCallback(async () => {
    try {
      const health = await fetchHealth(persona);
      setOverlay(health.overlay ?? null);
    } catch {
      // Bridge not reachable right now — keep what we had.
    }
  }, [persona]);

  useEffect(() => {
    void refreshOverlay();
  }, [refreshOverlay]);

  const check = useCallback(async () => {
    setState({ kind: "checking" });
    try {
      const res = await checkBrainUpdate();
      await refreshOverlay();
      if (res.available && res.commit && res.brain_version) {
        setState({ kind: "available", commit: res.commit, brainVersion: res.brain_version });
      } else if (res.reason === "unreachable" || res.reason === "bad_signature") {
        setState({ kind: "error", detail: `Couldn't check for a brain update: ${res.detail || res.reason}` });
      } else {
        setState({ kind: "none", reason: res.reason });
      }
    } catch (e) {
      setState({ kind: "error", detail: `Couldn't check for a brain update: ${errString(e) || "unknown error"}` });
    }
  }, [refreshOverlay]);

  const apply = useCallback(async () => {
    if (state.kind !== "available") return;
    setState({ kind: "applying", commit: state.commit });
    try {
      await applyBrainUpdate();
    } catch (e) {
      setState({ kind: "error", detail: `Couldn't update the brain: ${errString(e) || "unknown error"}` });
      return;
    }
    setState({ kind: "restarting" });
    if (await restart()) {
      await refreshOverlay();
      setState({ kind: "none", reason: "already_active" });
      return;
    }
    // §6: the updated bridge is unhealthy → roll back and restart; if that is
    // unhealthy too, the release brain and restart.
    try {
      await rollbackBrain("bridge unhealthy after a brain update");
    } catch {
      // fall through to the release brain
    }
    if (await restart()) {
      await refreshOverlay();
      setState({ kind: "rolled_back" });
      return;
    }
    try {
      await revertBrain();
    } catch {
      // the restart below reports the outcome
    }
    const ok = await restart();
    await refreshOverlay();
    setState(ok ? { kind: "rolled_back" } : { kind: "error", detail: RESTART_FAILED });
  }, [state, restart, refreshOverlay]);

  const useReleaseBrain = useCallback(async () => {
    setState({ kind: "reverting" });
    try {
      await revertBrain();
    } catch (e) {
      setState({ kind: "error", detail: `Couldn't switch to the release brain: ${errString(e) || "unknown error"}` });
      return;
    }
    const ok = await restart();
    await refreshOverlay();
    setState(ok ? { kind: "idle" } : { kind: "error", detail: RESTART_FAILED });
  }, [restart, refreshOverlay]);

  return { state, overlay, check, apply, useReleaseBrain };
}
