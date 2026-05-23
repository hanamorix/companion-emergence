import { useState, type ReactNode } from "react";
import { WButton, WizardShell } from "../components";
import type { MigrationReport, SkippedMemory } from "../../appConfig";

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

function tryParseMigrationReport(stdout: string): MigrationReport | null {
  const last = stdout.trim().split("\n").pop() ?? "";
  try {
    const obj = JSON.parse(last) as Record<string, unknown>;
    if (obj && obj.kind === "MigrationReport") return obj as unknown as MigrationReport;
  } catch {
    /* not JSON — fall through */
  }
  return null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(n: number): string {
  if (n >= 1_048_576) return `${(n / 1_048_576).toFixed(1)} MB`;
  if (n >= 1_024) return `${(n / 1_024).toFixed(1)} KB`;
  return `${n} B`;
}

// ---------------------------------------------------------------------------
// MigrationSummaryCard
// ---------------------------------------------------------------------------

const preStyle: React.CSSProperties = {
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
};

function MigrationSummaryCard({ report }: { report: MigrationReport }) {
  const isCopy = report.source_kind === "companion-emergence";
  return isCopy ? (
    <CopySummary report={report} />
  ) : (
    <MigrateSummary report={report} />
  );
}

function CopySummary({ report }: { report: MigrationReport }) {
  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 7,
        background: "rgba(40, 120, 60, 0.07)",
        border: "1px solid rgba(40, 120, 60, 0.35)",
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 500, color: "var(--text)", marginBottom: 6 }}>
        Transfer complete
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-mid)", lineHeight: 1.7 }}>
        <div>{formatBytes(report.bytes_copied)} copied across</div>
        {report.crystallizations_migrated > 0 && (
          <div>{report.crystallizations_migrated} crystallizations</div>
        )}
        <div style={{ marginTop: 6, fontStyle: "italic", color: "var(--text-mute)", fontSize: 10 }}>
          No migration needed — forward-copy.
        </div>
      </div>
    </div>
  );
}

function MigrateSummary({ report }: { report: MigrationReport }) {
  const [showSkipped, setShowSkipped] = useState(false);
  const skipped = report.memories_skipped;

  // Aggregate skip reasons
  const counts: Record<string, number> = {};
  for (const s of skipped) counts[s.reason] = (counts[s.reason] ?? 0) + 1;

  const visible = skipped.slice(0, 20);
  const overflow = skipped.length - visible.length;

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 7,
        background: "rgba(40, 120, 60, 0.07)",
        border: "1px solid rgba(40, 120, 60, 0.35)",
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 500, color: "var(--text)", marginBottom: 6 }}>
        Migration complete
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-mid)", lineHeight: 1.7 }}>
        <div>{report.memories_migrated} memories migrated</div>
        {report.crystallizations_migrated > 0 && (
          <div>{report.crystallizations_migrated} crystallizations migrated</div>
        )}
        {report.personality_copied && <div>Personality file copied</div>}
        {skipped.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div
              style={{
                fontSize: 10.5,
                color: "#a07434",
                fontFamily: "var(--font-disp)",
                textTransform: "uppercase",
                letterSpacing: "0.07em",
                marginBottom: 4,
                fontWeight: 500,
              }}
            >
              {skipped.length} skipped
            </div>
            {Object.entries(counts).map(([reason, count]) => (
              <div key={reason} style={{ paddingLeft: 8, fontSize: 10.5 }}>
                {reason}: {count}
              </div>
            ))}
            <button
              onClick={() => setShowSkipped((v) => !v)}
              style={{
                marginTop: 6,
                background: "none",
                border: "none",
                cursor: "pointer",
                fontSize: 10.5,
                color: "var(--accent)",
                padding: 0,
                fontFamily: "var(--font-ui)",
              }}
            >
              {showSkipped ? "Hide skipped IDs ▴" : "Show skipped IDs ▾"}
            </button>
            {showSkipped && (
              <div
                style={{
                  marginTop: 4,
                  paddingLeft: 8,
                  fontSize: 10.5,
                  fontFamily: "DM Mono, Courier New, monospace",
                  color: "var(--text-mid)",
                  lineHeight: 1.6,
                }}
              >
                {visible.map((s: SkippedMemory) => (
                  <div key={s.id}>
                    {s.id} — {s.reason}
                  </div>
                ))}
                {overflow > 0 && (
                  <div style={{ color: "var(--text-mute)" }}>+{overflow} more</div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

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
            ? "Your persona is ready. Opening Companion Emergence…"
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
          {result.ok ? (
            (() => {
              const report = tryParseMigrationReport(result.output);
              return report ? (
                <MigrationSummaryCard report={report} />
              ) : (
                <pre style={preStyle}>{result.output}</pre>
              );
            })()
          ) : (
            <pre style={preStyle}>{`${result.output}\n\n${result.error}`}</pre>
          )}
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
