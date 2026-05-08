import type { ReactNode } from "react";
import {
  Divider,
  FieldLabel,
  OptionCard,
  SectionLabel,
  WButton,
  WInput,
  WizardShell,
} from "../components";

export type MigrateSource = "nellbrain" | "emergence-kit";

interface Props {
  step: number;
  totalSteps: number;
  path: string;
  onPathChange: (p: string) => void;
  source: MigrateSource;
  onSourceChange: (s: MigrateSource) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepMigrate({
  step,
  totalSteps,
  path,
  onPathChange,
  source,
  onSourceChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  // Both sources require a path now — emergence-kit graduated from
  // "manual instructions" to "one-click import" once the auto-importer
  // landed.
  const valid = path.trim().length > 0;

  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Bring your brain over"
      subtitle="Tell us where your old companion lives. We'll carry their memories and personality across so the new framework starts already knowing who they are."
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
      <SectionLabel>Where are you coming from?</SectionLabel>
      <div role="radiogroup" aria-label="Migration source">
        <OptionCard
          selected={source === "nellbrain"}
          onClick={() => onSourceChange("nellbrain")}
          title="The original NellBrain framework"
          description="The bigger Python project with memories, soul crystallizations, connection-strength files, and a creative voice fingerprint."
        />
        <OptionCard
          selected={source === "emergence-kit"}
          onClick={() => onSourceChange("emergence-kit")}
          title="emergence-kit (or another simpler brain)"
          description="The lighter setup: my_brain.py + a few JSON files for memories, soul, and personality."
        />
      </div>

      <Divider />

      {source === "nellbrain" ? (
        <NellBrainPath path={path} onPathChange={onPathChange} />
      ) : (
        <EmergenceKitGuide path={path} onPathChange={onPathChange} />
      )}
    </WizardShell>
  );
}

function NellBrainPath({
  path,
  onPathChange,
}: {
  path: string;
  onPathChange: (p: string) => void;
}) {
  return (
    <>
      <SectionLabel>What carries over</SectionLabel>
      <Line emoji="🧠" title="Every memory" detail="how your companion remembers you and what mattered" />
      <Line emoji="🌱" title="Soul" detail="the moments they've crystallized as core to who they are" />
      <Line emoji="🕸️" title="Connections" detail="which memories light up together (Hebbian edges)" />
      <Line emoji="✨" title="Creative voice" detail="the writing-style fingerprint built up over time" />
      <Line emoji="📓" title="Recent journals" detail="reflex-arc entries that show their last few weeks" />
      <Line
        emoji="—"
        title="What you'll set fresh"
        detail="provider config, voice template, scheduling — picked in this wizard"
        muted
      />

      <div style={{ marginTop: 14 }}>
        <FieldLabel>Where is the original brain's data folder?</FieldLabel>
        <WInput
          label="Original brain data folder"
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
            lineHeight: 1.55,
          }}
        >
          The folder with <code>memories_v2.json</code>,{" "}
          <code>nell_soul.json</code>, and{" "}
          <code>connection_matrix.npy</code> inside. On most macOS
          installs this is <code>~/NellBrain/data</code>.
        </div>
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
          One safety check
        </div>
        <div style={{ fontSize: 11, color: "var(--text-mid)", lineHeight: 1.7 }}>
          <strong>Stop the original brain's bridge first</strong> if
          it's running. We won't touch your old files (the migration is
          read-only), but having two brains writing to the same memory
          file at the same time would corrupt it. From the original
          repo: <code>./bin/stop-bridge</code>, or kill the{" "}
          <code>nell_bridge</code> process. We'll check for the lock
          file and refuse if your old brain still looks active.
        </div>
      </div>
    </>
  );
}

function EmergenceKitGuide({
  path,
  onPathChange,
}: {
  path: string;
  onPathChange: (p: string) => void;
}) {
  return (
    <>
      <SectionLabel>What carries over</SectionLabel>
      <Line
        emoji="🧠"
        title="All memories"
        detail="memories_v2.json (or memories.json) → memories.db"
      />
      <Line
        emoji="🌱"
        title="Soul crystallizations"
        detail="entries from soul_template.json's crystallizations[]"
      />
      <Line
        emoji="📓"
        title="Personality file"
        detail="personality.json copied verbatim into the persona dir"
      />
      <Line
        emoji="—"
        title="What stays separate"
        detail="emergence-kit doesn't ship Hebbian edges, reflex arcs, creative DNA, or interests — those start fresh"
        muted
      />

      <Divider />

      <FieldLabel>Where is the emergence-kit folder?</FieldLabel>
      <WInput
        value={path}
        onChange={onPathChange}
        placeholder="/Users/you/emergence-kit"
        mono
      />
      <div
        style={{
          marginTop: 6,
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.55,
        }}
      >
        The folder containing <code>memories_v2.json</code> (or{" "}
        <code>memories.json</code>) and <code>soul_template.json</code>.
        On most installs this is the directory where you cloned{" "}
        <code>emergence-kit</code> and ran <code>my_brain.py</code>.
      </div>

      <div
        style={{
          marginTop: 14,
          padding: "10px 12px",
          borderRadius: 7,
          background: "rgba(60, 130, 90, 0.08)",
          border: "1px solid rgba(60, 130, 90, 0.40)",
          fontSize: 11,
          color: "var(--text-mid)",
          lineHeight: 1.7,
        }}
      >
        <strong style={{ color: "var(--text)" }}>One-click import is supported.</strong>{" "}
        Continue and we'll run{" "}
        <code>nell migrate --source emergence-kit</code> for you. Your
        old files are read-only — nothing in the emergence-kit folder
        will be touched.
      </div>

      <Divider />
      <div style={{ fontSize: 11, color: "var(--text-mid)", lineHeight: 1.65 }}>
        Whichever path you pick, your old files stay where they are.
        Nothing here touches the emergence-kit folder.
      </div>
    </>
  );
}

function Line({
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
        style={{ flexShrink: 0, fontSize: 13, width: 18, textAlign: "center" }}
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
