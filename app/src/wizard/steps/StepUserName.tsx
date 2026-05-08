import type { ReactNode } from "react";
import { FieldLabel, WButton, WInput, WizardShell } from "../components";

interface Props {
  step: number;
  totalSteps: number;
  value: string;
  onChange: (next: string) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepUserName({ step, totalSteps, value, onChange, onNext, onBack, avatar }: Props) {
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Your name"
      subtitle="Helps the brain tell you apart from people mentioned in memories and soul context."
      avatar={avatar}
      footer={
        <>
          <WButton variant="ghost" onClick={onBack} small>
            ← Back
          </WButton>
          <WButton onClick={onNext}>Continue →</WButton>
        </>
      }
    >
      <FieldLabel>Your name (optional)</FieldLabel>
      <WInput
        label="Your name"
        value={value}
        onChange={onChange}
        placeholder="e.g. Hana"
        onKeyDown={(e) => e.key === "Enter" && onNext()}
      />
      <div
        style={{
          marginTop: 6,
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.5,
        }}
      >
        Stored as <code style={{ background: "rgba(0,0,0,0.05)", padding: "0 3px", borderRadius: 3 }}>user_name</code> in
        persona_config.json. Only used locally for memory attribution.
      </div>
      <div
        style={{
          marginTop: 14,
          padding: "10px 12px",
          borderRadius: 7,
          background: "rgba(130,51,41,0.06)",
          border: "1px solid rgba(130,51,41,0.18)",
        }}
      >
        <div style={{ fontSize: 11.5, color: "var(--text)", lineHeight: 1.6 }}>
          <strong>Why this matters:</strong> the ingest extractor uses your name to distinguish
          you from historical figures the brain may reference in crystallisations. Without it,
          memories from your conversations may be mis-attributed.
        </div>
      </div>
    </WizardShell>
  );
}
