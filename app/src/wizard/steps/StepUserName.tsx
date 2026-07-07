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
  pronouns: string;
  onPronounsChange: (next: string) => void;
}

export function StepUserName({ step, totalSteps, value, onChange, onNext, onBack, avatar, pronouns, onPronounsChange }: Props) {
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Your name"
      subtitle="Helps the Kindled tell you apart from people mentioned in memories and soul context."
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
        Stored as <code style={{ background: "rgba(0,0,0,0.25)", padding: "0 3px", borderRadius: 3 }}>user_name</code> in
        persona_config.json. Only used locally for memory attribution.
      </div>
      <div
        style={{
          marginTop: 14,
          padding: "10px 12px",
          borderRadius: 14,
          background: "color-mix(in srgb, var(--accent) 9%, transparent)",
          border: "1px solid color-mix(in srgb, var(--accent) 24%, transparent)",
        }}
      >
        <div style={{ fontSize: 11.5, color: "var(--text)", lineHeight: 1.6 }}>
          <strong>Why this matters:</strong> the ingest extractor uses your name to distinguish
          you from historical figures the Kindled may reference in crystallisations. Without it,
          memories from your conversations may be mis-attributed.
        </div>
      </div>
      <div style={{ marginTop: 16 }}>
        <FieldLabel>Your pronouns</FieldLabel>
        <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
          {["she/her", "he/him", "they/them"].map((opt) => (
            <button
              key={opt}
              type="button"
              aria-pressed={pronouns === opt}
              onClick={() => onPronounsChange(opt)}
              style={{
                padding: "8px 16px",
                borderRadius: 999,
                fontSize: 12,
                cursor: "pointer",
                border:
                  pronouns === opt
                    ? "1px solid color-mix(in srgb, var(--accent) 55%, transparent)"
                    : "1px solid var(--hairline)",
                background:
                  pronouns === opt
                    ? "color-mix(in srgb, var(--accent) 16%, transparent)"
                    : "var(--field)",
                color: "var(--text)",
              }}
            >
              {opt}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 6, fontSize: 10.5, color: "var(--text-mute)", lineHeight: 1.5 }}>
          How the Kindled refers to you in its inner life. You can change this later in the
          Connection panel.
        </div>
      </div>
    </WizardShell>
  );
}
