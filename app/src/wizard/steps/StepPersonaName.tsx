import { useState, type ReactNode } from "react";
import {
  Divider,
  FieldError,
  FieldLabel,
  WButton,
  WInput,
  WizardShell,
} from "../components";

const PERSONA_RE = /^[A-Za-z0-9_-]{1,40}$/;

interface Props {
  step: number;
  totalSteps: number;
  value: string;
  onChange: (next: string) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepPersonaName({
  step,
  totalSteps,
  value,
  onChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  const [touched, setTouched] = useState(false);
  const valid = PERSONA_RE.test(value);
  const error =
    touched && value && !valid
      ? "Letters, digits, _ and - only · max 40 characters."
      : touched && !value
      ? "A persona name is required."
      : null;

  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Persona name"
      subtitle="This becomes the directory name — choose something you'll recognise."
      avatar={avatar}
      footer={
        <>
          <WButton variant="ghost" onClick={onBack} small>
            ← Back
          </WButton>
          <WButton onClick={onNext} disabled={!valid}>
            Continue →
          </WButton>
        </>
      }
    >
      <FieldLabel>Persona name</FieldLabel>
      <WInput
        label="Persona name"
        value={value}
        onChange={(v) => {
          onChange(v);
          setTouched(true);
        }}
        placeholder="e.g. nell or my_companion"
        mono
        error={!!error}
        maxLength={40}
        onKeyDown={(e) => e.key === "Enter" && valid && onNext()}
      />
      <FieldError msg={error} />
      {value && valid && (
        <div
          style={{
            marginTop: 10,
            padding: "8px 11px",
            borderRadius: 7,
            background: "rgba(130,51,41,0.07)",
            border: "1px solid rgba(130,51,41,0.18)",
          }}
        >
          <div
            style={{
              fontSize: 10,
              color: "var(--text-mute)",
              marginBottom: 3,
              fontFamily: "var(--font-disp)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Will be installed at
          </div>
          <div
            style={{
              fontSize: 11,
              color: "var(--text-mid)",
              fontFamily: "DM Mono, Courier New, monospace",
              wordBreak: "break-all",
            }}
          >
            $NELLBRAIN_HOME/personas/<strong style={{ color: "var(--accent)" }}>{value}</strong>/
          </div>
        </div>
      )}
      <Divider />
      <div style={{ fontSize: 10.5, color: "var(--text-mid)", lineHeight: 1.7 }}>
        <strong>Rules:</strong> letters, digits, <code>_</code> and <code>-</code> only — no
        spaces, slashes, or dots, max 40 characters. Run <code>nell status</code> after setup to
        confirm the path.
      </div>
    </WizardShell>
  );
}
