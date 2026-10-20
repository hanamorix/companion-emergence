/**
 * StopProgress (#456 Q-A, D18): while the bridge stops for a restart or at launch, what is still running and, 180 s
 * after the stop was requested, the hard-exit offer ("Hard exit" / "Keep waiting"). The same words as the terminal's
 * `nell supervisor stop`. Renders nothing when no stop is being watched.
 */

import { useRestartBridge } from "../../hooks/useRestartBridge";
import type { PersonaState } from "../../bridge";

interface Props {
  persona: string;
  currentMode: PersonaState["mode"];
}

const buttonStyle = {
  padding: "4px 10px",
  fontSize: 10.5,
  fontFamily: "var(--font-ui)",
  borderRadius: 6,
  cursor: "pointer",
} as const;

export function StopProgress({ persona, currentMode }: Props) {
  const { state, progress, offerHardExit, offerText, hardExit, keepWaiting } = useRestartBridge(
    persona,
    currentMode,
  );
  if (!progress || (state !== "requesting" && state !== "stopping" && state !== "starting")) return null;
  return (
    <div role="status" aria-live="polite" data-testid="stop-progress" style={{ marginTop: 8, fontSize: 10.5 }}>
      <div style={{ color: "var(--text-mid)", lineHeight: 1.45, wordBreak: "break-word" }}>{progress}</div>
      {offerHardExit && offerText && (
        <div style={{ marginTop: 6 }}>
          <div style={{ color: "var(--crimson)", lineHeight: 1.45, wordBreak: "break-word" }}>{offerText}</div>
          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <button
              type="button"
              onClick={hardExit}
              style={{
                ...buttonStyle,
                background: "rgba(178, 42, 42, 0.10)",
                color: "var(--crimson)",
                border: "1px solid rgba(178, 42, 42, 0.40)",
              }}
            >
              Hard exit
            </button>
            <button
              type="button"
              onClick={keepWaiting}
              style={{
                ...buttonStyle,
                background: "transparent",
                color: "var(--text-mid)",
                border: "1px solid rgba(234,222,218,0.25)",
              }}
            >
              Keep waiting
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
