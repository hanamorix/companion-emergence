/**
 * BrainUpdateRow — brain updates from main (#286 slice 4, spec §3.5). Sits under
 * the app-update row. Shows the running overlay (with "Use the release brain")
 * and, after "Check for updates", whether a signed brain-main build is available.
 */

import { useEffect } from "react";
import type { ReactNode } from "react";
import type { PersonaState } from "../../bridge";
import { useBrainUpdate } from "../../hooks/useBrainUpdate";

const muted = { fontSize: 11, color: "var(--text-mute)", padding: "6px 0", lineHeight: 1.45 } as const;
const buttonStyle = {
  width: "100%",
  padding: "7px 10px",
  fontSize: 11,
  fontFamily: "var(--font-ui)",
  color: "var(--text)",
  borderRadius: 6,
  cursor: "pointer",
} as const;

const short = (commit: string | null | undefined) => (commit ?? "").slice(0, 7);

export function BrainUpdateRow({
  persona,
  mode,
  checkToken,
}: {
  persona: string;
  mode: PersonaState["mode"];
  checkToken: number;
}) {
  const { state, overlay, check, apply, useReleaseBrain } = useBrainUpdate(persona, mode);

  useEffect(() => {
    if (checkToken > 0) void check();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- run once per token bump
  }, [checkToken]);

  const running =
    overlay && overlay.commit ? (
      <div style={{ marginTop: 8 }}>
        <div style={{ ...muted, color: "var(--text)" }}>
          Brain: main @{short(overlay.commit)}
          {overlay.brain_version ? ` (${overlay.brain_version})` : ""}
        </div>
        <button
          onClick={() => void useReleaseBrain()}
          disabled={state.kind === "applying" || state.kind === "restarting" || state.kind === "reverting"}
          style={{
            ...buttonStyle,
            background: "transparent",
            border: "1px solid color-mix(in srgb, var(--text-mute) 40%, transparent)",
          }}
        >
          Use the release brain
        </button>
      </div>
    ) : null;

  let status: ReactNode = null;
  switch (state.kind) {
    case "checking":
      status = <div style={muted}>Checking for a brain update…</div>;
      break;
    case "available":
      status = (
        <div style={{ marginTop: 8 }}>
          <div style={{ ...muted, color: "var(--text)" }}>
            Brain update available: main @{short(state.commit)} ({state.brainVersion})
          </div>
          <button
            onClick={() => void apply()}
            style={{
              ...buttonStyle,
              background: "rgba(79,168,118,0.15)",
              border: "1px solid rgba(79,168,118,0.45)",
            }}
          >
            End conversation and update the brain
          </button>
        </div>
      );
      break;
    case "applying":
      status = <div style={muted}>Installing the brain update…</div>;
      break;
    case "restarting":
      status = <div style={muted}>Restarting the brain…</div>;
      break;
    case "reverting":
      status = <div style={muted}>Switching to the release brain…</div>;
      break;
    case "none":
      if (state.reason === "dev_build") break;
      status = (
        <div style={muted}>
          {state.reason === "app_too_old"
            ? "A brain update needs a newer app — update Companion Emergence first."
            : "The brain is up to date ✓"}
        </div>
      );
      break;
    case "rolled_back":
      status = (
        <div role="status" style={{ ...muted, color: "#a07434" }}>
          The new brain didn't start, so it was rolled back.
        </div>
      );
      break;
    case "error":
      status = (
        <div style={{ ...muted, color: "var(--crimson)", wordBreak: "break-word" }}>
          {state.detail}
        </div>
      );
      break;
  }

  if (!running && !status) return null;
  return (
    <div>
      {status}
      {running}
    </div>
  );
}
