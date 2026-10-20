/**
 * RestartBridgeButton — embedded inside StatusBanner (only when
 * kind === "error") so the recovery action lives where the failure is
 * already visible. Owns no state; defers to useRestartBridge for the
 * full restart choreography.
 *
 * Spec: docs/superpowers/specs/2026-05-17-bridge-restart-button-design.md
 * §3 (frontend), §5 (labels), §6 (re-entry guard).
 */

import {
  useRestartBridge,
  type RestartState,
} from "../../hooks/useRestartBridge";
import type { PersonaState } from "../../bridge";
import { StopProgress } from "./StopProgress";

interface Props {
  persona: string;
  currentMode: PersonaState["mode"];
}

const LABELS: Record<RestartState, string> = {
  idle: "Restart the brain",
  requesting: "Asking the brain to stop…",
  stopping: "Waiting for the brain to finish…",
  forcing: "The brain is not answering, ending it…",
  starting: "Starting the brain…",
  waiting_for_health: "Waiting for the brain to come back…",
  reconnecting: "Reconnecting…",
  success: "Restarted ✓",
  failed: "Retry",
};

export function RestartBridgeButton({ persona, currentMode }: Props) {
  const { state, errorDetail, fellThrough, restart } = useRestartBridge(persona, currentMode);
  const interactive = state === "idle" || state === "failed";

  return (
    <div style={{ marginTop: 8 }}>
      <button
        type="button"
        onClick={interactive ? restart : undefined}
        disabled={!interactive}
        aria-live="polite"
        style={{
          width: "100%",
          padding: "6px 10px",
          fontSize: 10.5,
          fontFamily: "var(--font-ui)",
          background: "rgba(178, 42, 42, 0.10)",
          color: "var(--crimson)",
          border: "1px solid rgba(178, 42, 42, 0.40)",
          borderRadius: 6,
          cursor: interactive ? "pointer" : "wait",
          opacity: interactive ? 1 : 0.75,
        }}
      >
        {LABELS[state]}
      </button>
      <StopProgress persona={persona} currentMode={currentMode} />
      {(state === "forcing" || state === "failed") && fellThrough && (
        <div
          style={{
            fontSize: 10,
            color: "var(--text-mute)",
            marginTop: 6,
            lineHeight: 1.45,
            wordBreak: "break-word",
          }}
        >
          Had to end the brain: {fellThrough}
        </div>
      )}
      {state === "failed" && errorDetail && (
        <div
          style={{
            fontSize: 10,
            color: "var(--crimson)",
            marginTop: 6,
            lineHeight: 1.45,
            wordBreak: "break-word",
          }}
        >
          {errorDetail}
        </div>
      )}
    </div>
  );
}
