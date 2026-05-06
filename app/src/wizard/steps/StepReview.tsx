import { Divider, SectionLabel, WButton, WizardShell } from "../components";
import type { WizardState } from "../Wizard";

import type { ReactNode } from "react";

interface Props {
  step: number;
  totalSteps: number;
  state: WizardState;
  onInstall: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepReview({ step, totalSteps, state, onInstall, onBack, avatar }: Props) {
  const equivalentCli = buildEquivalentCli(state);
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Review & confirm"
      subtitle="Everything looks right? Confirm to run the installation."
      avatar={avatar}
      footer={
        <>
          <WButton variant="ghost" onClick={onBack} small>
            ← Back
          </WButton>
          <WButton onClick={onInstall}>Install →</WButton>
        </>
      }
    >
      <SectionLabel>Configuration</SectionLabel>
      <Row label="Mode" value={state.mode === "fresh" ? "Fresh brain" : "Migrate"} />
      <Row label="Persona name" value={state.personaName} mono />
      <Row label="Your name" value={state.userName || "(unset)"} mono />
      <Row label="Provider" value={state.provider} mono />
      <Row label="Voice template" value={state.voiceTemplate} mono />
      {state.mode === "migrate" && (
        <Row label="OG data path" value={state.migrateFromPath} mono small />
      )}
      <Divider />
      <SectionLabel>Equivalent CLI command</SectionLabel>
      <pre
        style={{
          fontSize: 10.5,
          fontFamily: "DM Mono, Courier New, monospace",
          background: "rgba(0,0,0,0.05)",
          color: "var(--text-mid)",
          padding: 10,
          borderRadius: 6,
          margin: 0,
          whiteSpace: "pre-wrap",
          lineHeight: 1.5,
        }}
      >
        {equivalentCli}
      </pre>
    </WizardShell>
  );
}

function Row({
  label,
  value,
  mono,
  small,
}: {
  label: string;
  value: string;
  mono?: boolean;
  small?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        gap: 12,
        padding: "5px 0",
        borderBottom: "1px solid rgba(191,184,173,0.4)",
        fontSize: small ? 10.5 : 12,
      }}
    >
      <div style={{ color: "var(--text-mid)" }}>{label}</div>
      <div
        style={{
          color: "var(--text)",
          fontWeight: 500,
          fontFamily: mono ? "DM Mono, Courier New, monospace" : "var(--font-ui)",
          textAlign: "right",
          wordBreak: "break-all",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function buildEquivalentCli(s: WizardState): string {
  const lines = ["uv run nell init \\"];
  lines.push(`  --persona ${s.personaName} \\`);
  if (s.userName.trim()) lines.push(`  --user-name "${s.userName}" \\`);
  lines.push(`  --provider ${s.provider} \\`);
  lines.push(`  --voice-template ${s.voiceTemplate}${s.mode === "migrate" ? " \\" : ""}`);
  if (s.mode === "migrate") {
    lines.push(`  --migrate-from ${s.migrateFromPath}`);
  } else {
    lines[lines.length - 1] += " \\";
    lines.push("  --fresh");
  }
  return lines.join("\n");
}

