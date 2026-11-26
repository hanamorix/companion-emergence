/**
 * StopProgress (#456 Q-A, D18): while the bridge stops for a restart or at launch, what is still running and, 180 s
 * after the stop was requested, the hard-exit offer ("Hard exit" / "Keep waiting"). The same words as the terminal's
 * `nell supervisor stop`. Renders nothing when no stop is being watched.
 *
 * Shown wherever the user is (review S9): the app renders one floating instance over the main screen (every restart
 * flow: the Restart button, the brain update, the login prompt, the version replace, the supervisor install) and one
 * on the boot screen (the launch check and the memory-recovery restart).
 */

import { useRestartBridge } from "../../hooks/useRestartBridge";
import type { PersonaState } from "../../bridge";

interface Props {
  persona: string;
  currentMode: PersonaState["mode"];
  /** A card floating over the main screen instead of an inline block. */
  floating?: boolean;
}

const floatingStyle = {
  position: "fixed",
  top: 14,
  left: "50%",
  transform: "translateX(-50%)",
  width: "min(440px, calc(100vw - 32px))",
  zIndex: 50,
  padding: "10px 12px",
  borderRadius: 8,
  border: "1px solid var(--panel-border, rgba(234,222,218,0.18))",
  background: "var(--panel-bg)",
  boxShadow: "0 6px 24px rgba(0,0,0,0.25)",
} as const;

const buttonStyle = {
  padding: "4px 10px",
  fontSize: 10.5,
  fontFamily: "var(--font-ui)",
  borderRadius: 6,
  cursor: "pointer",
} as const;

export function StopProgress({ persona, currentMode, floating = false }: Props) {
  const { state, progress, offerHardExit, offerText, hardExit, keepWaiting } = useRestartBridge(
    persona,
    currentMode,
  );
  if (!progress || (state !== "requesting" && state !== "stopping" && state !== "starting")) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="stop-progress"
      style={floating ? { ...floatingStyle, fontSize: 11 } : { marginTop: 8, fontSize: 10.5 }}
    >
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
