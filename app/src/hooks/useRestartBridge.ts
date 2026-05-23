/**
 * useRestartBridge — orchestration hook for the v0.0.14 manual bridge
 * restart flow. Drives the state machine:
 *
 *   idle → closing → shutting_down → waiting_for_health → reconnecting → success
 *
 * with a `forcing` branch reachable from any of the first three states
 * when the matching HTTP step times out. A second failure after `forcing`
 * lands in `failed` with a user-readable error string.
 *
 * Spec: docs/superpowers/specs/2026-05-17-bridge-restart-button-design.md
 * Plan: docs/superpowers/plans/2026-05-17-bridge-restart-button.md (Phase 3).
 *
 * Timeouts are spec-locked (§5):
 *   - /sessions/close response:    5s
 *   - /supervisor/shutdown 202:    3s
 *   - /health polling window:     30s (per attempt; two attempts max)
 *   - post-SIGKILL grace:         handled inside Tauri force_restart_bridge
 *
 * The hook owns network calls + state; the parent component renders the
 * button and watches `state`. Parent must call `onModeChanged(mode)`
 * (or pass the current mode as the second arg) so reconnecting → success
 * lands when /state poll flips back to "live".
 */

import { useState, useCallback, useEffect, useRef } from "react";
import type { PersonaState } from "../bridge";
import {
  closeActiveSession,
  shutdownBridge,
  invokeForceRestart,
  fetchHealth,
} from "../bridge";
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
  restart: () => void;
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

export function useRestartBridge(
  persona: string,
  currentMode: PersonaState["mode"],
): UseRestartBridge {
  const [state, setState] = useState<RestartState>("idle");
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  // Tracks the latest reachable state without going through React's render
  // queue — needed inside the async restart() body where stale closures
  // would otherwise see "idle" forever.
  const stateRef = useRef<RestartState>("idle");
  const inFlightRef = useRef<boolean>(false);

  const transition = useCallback((next: RestartState) => {
    stateRef.current = next;
    setState(next);
  }, []);

  const restart = useCallback(() => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setErrorDetail(null);

    const run = async () => {
      try {
        // Try graceful close → shutdown → health. Any timeout escalates
        // to SIGKILL fallback. A second health-poll failure → failed.
        const tryGraceful = async (): Promise<boolean> => {
          transition("closing");
          try {
            await withTimeout(
              closeActiveSession(persona),
              TIMEOUT_CLOSE_MS,
              "/sessions/close",
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
            // proceed to health poll.
          }

          transition("waiting_for_health");
          try {
            await pollHealth(persona, Date.now() + TIMEOUT_HEALTH_MS);
          } catch {
            return false;
          }
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
          return;
        }
        transition("reconnecting");
      } finally {
        inFlightRef.current = false;
      }
    };

    void run();
  }, [persona, transition]);

  const onModeChanged = useCallback(
    (mode: PersonaState["mode"]) => {
      if (stateRef.current === "reconnecting" && mode === "live") {
        transition("success");
      }
    },
    [transition],
  );

  // Also catch the live-mode flip via the prop, so parents that only
  // re-render (without explicitly calling onModeChanged) still resolve
  // the terminal state.
  useEffect(() => {
    if (state === "reconnecting" && currentMode === "live") {
      transition("success");
    }
  }, [currentMode, state, transition]);

  return { state, errorDetail, restart, onModeChanged };
}
