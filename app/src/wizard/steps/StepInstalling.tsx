import type { ReactNode } from "react";
import { WButton, WizardShell } from "../components";

interface Props {
  step: number;
  totalSteps: number;
  result: { ok: boolean; output: string; error: string } | null;
  onRetry: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepInstalling({ step, totalSteps, result, onRetry, onBack, avatar }: Props) {
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title={result ? (result.ok ? "Installed" : "Install failed") : "Installing…"}
      subtitle={
        result
          ? result.ok
            ? "Your persona is ready. Opening NellFace…"
            : "The migrator returned a non-zero exit. Inspect the error and retry."
          : "Running uv run nell init — this may take a minute for migration runs."
      }
      avatar={avatar}
      footer={
        result && !result.ok ? (
          <>
            <WButton variant="ghost" onClick={onBack} small>
              ← Back
            </WButton>
            <WButton onClick={onRetry}>Retry →</WButton>
          </>
        ) : (
          <span
            style={{
              fontSize: 10.5,
              color: "var(--text-mute)",
              fontFamily: "var(--font-disp)",
              fontStyle: "italic",
            }}
          >
            {result?.ok ? "✓ done" : "working…"}
          </span>
        )
      }
    >
      {!result && <Spinner />}
      {result && (
        <div>
          <pre
            style={{
              fontSize: 10.5,
              fontFamily: "DM Mono, Courier New, monospace",
              background: "rgba(0,0,0,0.05)",
              color: "var(--text-mid)",
              padding: 10,
              borderRadius: 6,
              margin: 0,
              maxHeight: 220,
              overflowY: "auto",
              whiteSpace: "pre-wrap",
              lineHeight: 1.5,
            }}
          >
            {result.ok ? result.output : `${result.output}\n\n${result.error}`}
          </pre>
        </div>
      )}
    </WizardShell>
  );
}

function Spinner() {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "30px 0" }}>
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: "50%",
          border: "2px solid rgba(130,51,41,0.2)",
          borderTopColor: "var(--accent)",
          animation: "spin 0.9s linear infinite",
        }}
      />
    </div>
  );
}
