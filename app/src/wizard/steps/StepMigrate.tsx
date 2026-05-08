import type { ReactNode } from "react";
import { Divider, FieldLabel, SectionLabel, WButton, WInput, WizardShell } from "../components";

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
      title="Migrate from the original framework"
      subtitle="Point us at the original brain's data folder and we'll port the memories, soul, Hebbian edges, and creative DNA into the new layout."
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
      <SectionLabel>What gets migrated</SectionLabel>
      <ListLine emoji="🧠" title="Memories" detail="memories_v2.json → memories.db with embeddings rebuilt" />
      <ListLine emoji="🌱" title="Soul" detail="nell_soul.json crystallizations and resonance scores" />
      <ListLine emoji="🕸️" title="Hebbian edges" detail="connection_matrix.npy → hebbian.db" />
      <ListLine emoji="✨" title="Creative DNA" detail="creative_dna.json (writing voice fingerprint)" />
      <ListLine emoji="📓" title="Reflex log" detail="recent journal-arc fires retained for trajectory" />
      <ListLine
        emoji="—"
        title="Stays separate"
        detail="provider config, voice template, daemon cron schedules — re-set in this wizard"
        muted
      />

      <Divider />

      <FieldLabel>Original brain data directory</FieldLabel>
      <WInput
        value={path}
        onChange={onPathChange}
        placeholder="/Users/you/NellBrain/data"
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
        The folder containing <code>memories_v2.json</code>, <code>nell_soul.json</code>,
        <code> connection_matrix.npy</code>. On a typical macOS install
        the path is <code>~/NellBrain/data</code>.
      </div>

      <div
        style={{
          marginTop: 14,
          padding: "10px 12px",
          borderRadius: 7,
          background: "rgba(216,154,88,0.10)",
          border: "1px solid rgba(216,154,88,0.40)",
        }}
      >
        <div
          style={{
            fontSize: 10.5,
            color: "#a07434",
            fontFamily: "var(--font-disp)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            marginBottom: 6,
            fontWeight: 500,
          }}
        >
          Before you continue
        </div>
        <div style={{ fontSize: 11, color: "var(--text-mid)", lineHeight: 1.7 }}>
          <strong>Stop the original brain's bridge first.</strong> The
          migrator checks <code>memories_v2.json.lock</code> and refuses
          if the OG bridge has touched it within the last 90 seconds —
          this prevents a half-written migration. From the original
          repo: <code>./bin/stop-bridge</code>, or kill the
          <code> nell_bridge</code> process. The migration is read-only
          on the original data.
        </div>
      </div>
    </WizardShell>
  );
}

function ListLine({
  emoji,
  title,
  detail,
  muted,
}: {
  emoji: string;
  title: string;
  detail: string;
  muted?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 8,
        padding: "5px 0",
        opacity: muted ? 0.7 : 1,
      }}
    >
      <span
        aria-hidden="true"
        style={{
          flexShrink: 0,
          fontSize: 13,
          width: 18,
          textAlign: "center",
        }}
      >
        {emoji}
      </span>
      <div style={{ fontSize: 11.5, lineHeight: 1.5 }}>
        <span style={{ fontWeight: 500, color: "var(--text)" }}>{title}</span>
        <span style={{ color: "var(--text-mid)" }}> — {detail}</span>
      </div>
    </div>
  );
}
