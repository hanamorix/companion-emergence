import { useState } from "react";
import { StepRecover } from "../../wizard/steps/StepRecover";
import { SectionLabel } from "../ui";

/**
 * Recover memories — reachable entry point from the Connection panel.
 *
 * The wizard's recover step is only reached during a migrate flow, which a
 * working single-persona user never sees at boot. This surfaces the same
 * StepRecover from the diagnostics panel so a user whose memories/links went
 * missing after a transfer can restore them: from their original source
 * folder, or (blank) from this Kindled's own graveyard.
 */
export function RecoverLauncher({ persona }: { persona: string }) {
  const [active, setActive] = useState(false);
  const [source, setSource] = useState("");
  const [sourceDir, setSourceDir] = useState<string | null>(null);

  if (active) {
    return (
      <div>
        <SectionLabel>Recover memories</SectionLabel>
        <StepRecover persona={persona} sourceDir={sourceDir} onDone={() => setActive(false)} />
        <button
          onClick={() => setActive(false)}
          style={{
            marginTop: 8,
            background: "none",
            border: "none",
            padding: 0,
            color: "var(--text-mute)",
            cursor: "pointer",
            fontSize: 10.5,
            fontFamily: "var(--font-ui)",
            textDecoration: "underline",
          }}
        >
          cancel
        </button>
      </div>
    );
  }

  return (
    <div>
      <SectionLabel>Recover memories</SectionLabel>
      <div
        style={{
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.55,
          marginBottom: 8,
          letterSpacing: "0.01em",
        }}
      >
        If memories or links went missing after a transfer, restore them from your
        original persona folder — or leave the path blank to recover from this
        Kindled&rsquo;s own graveyard.
      </div>
      <input
        value={source}
        onChange={(e) => setSource(e.target.value)}
        placeholder="path to original persona folder (optional)"
        spellCheck={false}
        style={{
          width: "100%",
          boxSizing: "border-box",
          padding: "6px 8px",
          marginBottom: 8,
          fontSize: 11,
          fontFamily: "var(--font-disp)",
          background: "var(--bg-input, rgba(0,0,0,0.18))",
          color: "var(--text)",
          border: "1px solid rgba(130, 51, 41, 0.30)",
          borderRadius: 6,
        }}
      />
      <button
        onClick={() => {
          setSourceDir(source.trim() || null);
          setActive(true);
        }}
        style={{
          width: "100%",
          padding: "7px 10px",
          fontSize: 11,
          fontFamily: "var(--font-ui)",
          background: "var(--accent-dim)",
          color: "var(--text)",
          border: "1px solid rgba(130, 51, 41, 0.30)",
          borderRadius: 6,
          cursor: "pointer",
        }}
      >
        Recover memories…
      </button>
    </div>
  );
}
