/**
 * useRestartBridge — orchestration hook for the v0.0.14 manual bridge
 * restart flow. Drives the state machine:
 *
 *   idle → closing → shutting_down → reconnecting → waiting_for_health → success
 *
 * with a `forcing` branch reachable from any of the first three states
 * when the matching HTTP step times out. A second failure after `forcing`
 * lands in `failed` with a user-readable error string.
 *
 * Task 6 reorder: graceful path now calls ensureBridgeRunning between
 * shutdown and health poll, and transitions reconnecting → waiting_for_health
 * itself. The forced path still transitions waiting_for_health internally
 * and leaves the outer reconnecting transition to fire after.
 *
 * Spec: docs/superpowers/specs/2026-05-17-bridge-restart-button-design.md
 * Plan: docs/superpowers/plans/2026-05-17-bridge-restart-button.md (Phase 3).
 *
 * Timeouts are spec-locked (§5):
 *   - /sessions/snapshot response: 5s
 *   - /supervisor/shutdown 202:    3s
 *   - /health polling window:     30s (per attempt; two attempts max)
 *   - post-SIGKILL grace:         handled inside Tauri force_restart_bridge
 *
 * The hook owns network calls + state; the parent component renders the
 * button and watches `state`. Parent must call `onModeChanged(mode)`
 * (or pass the current mode as the second arg) so reconnecting → success
 * lands when /state poll flips back to "live".
 *
 * `restart()` resolves true when the bridge is healthy again (graceful or
 * forced path); false when the flow ends in `failed`.
 *
 * One restart app-wide (#310): App, RestartBridgeButton and useBrainUpdate each
 * hold an instance, so the state and the in-flight run live at module scope. A
 * caller that asks while a restart is running joins it and gets its result,
 * and every instance renders the same state (the button stays disabled while
 * a brain update's restart runs).
 */

import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { PersonaState } from "../bridge";
import {
  snapshotActiveSession,
  shutdownBridge,
  invokeForceRestart,
  fetchHealth,
} from "../bridge";
import { ensureBridgeRunning } from "../appConfig";
import { errString } from "../lib/errString";

export type RestartState =
  | "idle"
  | "closing"
  | "shutting_down"
  | "waiting_for_health"
  | "forcing"
  | "reconnecting"
  | "success"
  | "failed";

export interface UseRestartBridge {
  state: RestartState;
  errorDetail: string | null;
  restart: () => Promise<boolean>;
  onModeChanged: (mode: PersonaState["mode"]) => void;
}

// Spec-locked timeouts (§5). Exported so component labels and tests
// can reason about them without re-deriving the values.
export const TIMEOUT_CLOSE_MS = 5000;
export const TIMEOUT_SHUTDOWN_MS = 3000;
export const TIMEOUT_HEALTH_MS = 30000;
export const HEALTH_POLL_INTERVAL_MS = 500;

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const handle = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)),
      ms,
    );
    p.then(
      (v) => {
        clearTimeout(handle);
        resolve(v);
      },
      (e) => {
        clearTimeout(handle);
        reject(e);
      },
    );
  });
}

async function pollHealth(persona: string, deadline: number): Promise<void> {
  while (Date.now() < deadline) {
    try {
      await fetchHealth(persona);
      return;
    } catch {
      // Bridge isn't up yet; wait and retry.
      await new Promise((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
    }
  }
  throw new Error("/health poll window expired");
}

const FAILED_USER_MESSAGE =
  "Restart failed. Try `nell service status` from terminal, or restart Companion Emergence.";

// Module scope (#310), same pattern as useBrainUpdate: the bridge is app-wide,
// so its restart is too. `inFlight` is the running restart, which later callers
// join.
// ponytail: one bridge per app, so one store; key by persona if that changes.
let shared: { state: RestartState; errorDetail: string | null } = {
  state: "idle",
  errorDetail: null,
};
let inFlight: Promise<boolean> | null = null;
const listeners = new Set<() => void>();

function setShared(patch: Partial<typeof shared>): void {
  shared = { ...shared, ...patch };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function transition(next: RestartState): void {
  setShared({ state: next });
}

function setErrorDetail(detail: string | null): void {
  setShared({ errorDetail: detail });
}

/** Back to idle after a restart nobody pressed (the launch path, #335), so the
 *  Restart button doesn't show its result. A no-op while a restart runs. */
export function clearRestartState(): void {
  if (!inFlight) setShared({ state: "idle", errorDetail: null });
}

/** Reset for test isolation — never call in production code. */
export function _resetRestartBridgeForTests(): void {
  shared = { state: "idle", errorDetail: null };
  inFlight = null;
}

/** The app-wide restart (#310). Plain function so code that runs before any
 *  component mounts (the launch path, #335) can call it; the hook wraps it. */
export function restartBridge(persona: string): Promise<boolean> {
  if (inFlight) return inFlight;
  setErrorDetail(null);

  const run = async (): Promise<boolean> => {
    try {
      // Try graceful snapshot → shutdown → ensureBridgeRunning → health.
      // Any timeout on snapshot escalates to SIGKILL fallback.
      // A second health-poll failure → failed.
      const tryGraceful = async (): Promise<boolean> => {
        transition("closing");
        try {
          await withTimeout(
            snapshotActiveSession(persona),
            TIMEOUT_CLOSE_MS,
            "/sessions/snapshot",
          );
        } catch {
          return false;
        }

        transition("shutting_down");
        try {
          await withTimeout(
            shutdownBridge(persona),
            TIMEOUT_SHUTDOWN_MS,
            "/supervisor/shutdown",
          );
        } catch {
          // Network failures here are expected — the bridge drops the
          // connection as it dies. Spec §6.4: treat as success and
          // proceed to ensureBridgeRunning + health poll.
        }

        // Explicitly start a fresh bridge so the health poll has something
        // to reach. Failure here means we can't guarantee a live bridge —
        // escalate to forced restart.
        transition("reconnecting");
        try {
          await ensureBridgeRunning(persona);
        } catch {
          return false;
        }

        transition("waiting_for_health");
        try {
          await pollHealth(persona, Date.now() + TIMEOUT_HEALTH_MS);
        } catch {
          return false;
        }
        // Bridge is healthy — move to reconnecting so the parent's
        // live-mode flip (onModeChanged / prop-effect) can resolve to success.
        transition("reconnecting");
        return true;
      };

      const tryForced = async (): Promise<boolean> => {
        transition("forcing");
        try {
          await invokeForceRestart(persona);
        } catch (e) {
          setErrorDetail(errString(e) || FAILED_USER_MESSAGE);
          return false;
        }
        transition("waiting_for_health");
        try {
          await pollHealth(persona, Date.now() + TIMEOUT_HEALTH_MS);
        } catch {
          setErrorDetail(FAILED_USER_MESSAGE);
          return false;
        }
        return true;
      };

      const gracefulOk = await tryGraceful();
      const ok = gracefulOk || (await tryForced());
      if (!ok) {
        transition("failed");
        return false;
      }
      // Graceful path ends in reconnecting (after ensureBridgeRunning +
      // pollHealth — the final reconnecting is the handoff to the parent's
      // live-mode flip). Forced path ends in waiting_for_health; push it
      // to reconnecting here so onModeChanged / prop-effect can resolve
      // reconnecting → success when mode flips live.
      if (!gracefulOk) {
        transition("reconnecting");
      }
      return true;
    } finally {
      inFlight = null;
    }
  };

  inFlight = run();
  return inFlight;
}

export function useRestartBridge(
  persona: string,
  currentMode: PersonaState["mode"],
): UseRestartBridge {
  const { state, errorDetail } = useSyncExternalStore(subscribe, () => shared);

  const restart = useCallback(() => restartBridge(persona), [persona]);

  const onModeChanged = useCallback((mode: PersonaState["mode"]) => {
    if (shared.state === "reconnecting" && mode === "live") {
      transition("success");
    }
  }, []);

  // Also catch the live-mode flip via the prop, so parents that only
  // re-render (without explicitly calling onModeChanged) still resolve
  // the terminal state. And once the bridge drops again, "success" belongs
  // to the last incident: back to idle so the banner's button is clickable
  // (per-instance state used to get this from the banner remounting).
  useEffect(() => {
    if (state === "reconnecting" && currentMode === "live") {
      transition("success");
    } else if (state === "success" && currentMode !== "live") {
      transition("idle");
    }
  }, [currentMode, state]);

  return { state, errorDetail, restart, onModeChanged };
}
