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

interface Props {
  persona: string;
  currentMode: PersonaState["mode"];
}

const LABELS: Record<RestartState, string> = {
  idle: "End conversation and restart",
  closing: "Ending conversation…",
  shutting_down: "Shutting bridge down…",
  waiting_for_health: "Waiting for bridge to come back…",
  forcing: "Bridge not responding — forcing restart…",
  reconnecting: "Reconnecting…",
  success: "Restarted ✓",
  failed: "Retry",
};

export function RestartBridgeButton({ persona, currentMode }: Props) {
  const { state, errorDetail, restart } = useRestartBridge(persona, currentMode);
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
