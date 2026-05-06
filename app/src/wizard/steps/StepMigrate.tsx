import type { ReactNode } from "react";
import { FieldLabel, WButton, WInput, WizardShell } from "../components";

interface Props {
  step: number;
  totalSteps: number;
  path: string;
  onPathChange: (p: string) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepMigrate({
  step,
  totalSteps,
  path,
  onPathChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  const valid = path.trim().length > 0;
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Migrate original data"
      subtitle="Port your existing brain's memories, soul, and edges to the new framework."
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
      <FieldLabel>Original Brain data directory</FieldLabel>
      <WInput
        value={path}
        onChange={onPathChange}
        placeholder="/path/to/OriginalBrain/data"
        mono
      />
      <div
        style={{
          marginTop: 6,
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.5,
        }}
      >
        The directory containing <code>memories_v2.json</code>, <code>nell_soul.json</code>,{" "}
        <code>connection_matrix.npy</code>, etc.
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
        <div
          style={{
            fontSize: 10.5,
            color: "var(--text-mute)",
            fontFamily: "var(--font-disp)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            marginBottom: 6,
          }}
        >
          Prerequisites
        </div>
        <div style={{ fontSize: 11, color: "var(--text-mid)", lineHeight: 1.7 }}>
          <strong>Original Brain bridge is not running.</strong> The migrator checks{" "}
          <code>memories_v2.json.lock</code> — it will refuse if the OG bridge looks active
          (90s threshold). Pass <code>--force-preflight</code> from the CLI if you're sure
          the lock is stale.
        </div>
      </div>
    </WizardShell>
  );
}
