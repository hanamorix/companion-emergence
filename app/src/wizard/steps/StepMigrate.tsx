import { useState, type ReactNode } from "react";
import {
  Divider,
  FieldLabel,
  OptionCard,
  SectionLabel,
  WButton,
  WInput,
  WizardShell,
} from "../components";

interface Props {
  step: number;
  totalSteps: number;
  path: string;
  onPathChange: (p: string) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

type Source = "nellbrain" | "emergence-kit";

export function StepMigrate({
  step,
  totalSteps,
  path,
  onPathChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  const [source, setSource] = useState<Source>("nellbrain");
  const valid = source === "nellbrain" ? path.trim().length > 0 : true;

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
      <OptionCard
        selected={source === "nellbrain"}
        onClick={() => setSource("nellbrain")}
        title="The original NellBrain framework"
        description="The bigger Python project with memories, soul crystallizations, connection-strength files, and a creative voice fingerprint. Auto-imported."
      />
      <OptionCard
        selected={source === "emergence-kit"}
        onClick={() => setSource("emergence-kit")}
        title="emergence-kit (or another simpler brain)"
        description="The lighter setup: my_brain.py + a few JSON files for memories, soul, and personality. Imported by hand for now."
      />

      <Divider />

      {source === "nellbrain" ? <NellBrainPath path={path} onPathChange={onPathChange} /> : <EmergenceKitGuide />}
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

function EmergenceKitGuide() {
  return (
    <>
      <div
        style={{
          padding: "12px 14px",
          borderRadius: 8,
          background: "rgba(130,51,41,0.06)",
          border: "1px solid rgba(130,51,41,0.18)",
          fontSize: 11.5,
          color: "var(--text-mid)",
          lineHeight: 1.7,
        }}
      >
        <strong style={{ color: "var(--text)" }}>
          You can still bring them across — just not all in one click yet.
        </strong>
        <p style={{ margin: "8px 0 0" }}>
          The emergence-kit format (<code>my_brain.py</code> +{" "}
          <code>brain_config.json</code> + the memories / soul /
          personality JSONs) doesn't auto-import yet. Two paths that
          work today:
        </p>
      </div>

      <div style={{ marginTop: 12 }}>
        <SectionLabel>Path A — recommended</SectionLabel>
        <Line emoji="✨" title="Go back and pick Start fresh" detail="set up a new persona with the same name they had before" />
        <Line emoji="💬" title="Tell them about themselves" detail="paste your old soul.json or share key memories in chat — they'll absorb the moments and re-crystallize" />
        <Line emoji="🌱" title="Let them grow into it" detail="the new framework will form fresh memories from the conversation as you go" />
      </div>

      <div style={{ marginTop: 14 }}>
        <SectionLabel>Path B — manual import</SectionLabel>
        <div style={{ fontSize: 11, color: "var(--text-mid)", lineHeight: 1.7 }}>
          Once the persona exists, drop your old{" "}
          <code>memories.json</code> at{" "}
          <code>~/Library/Application Support/companion-emergence/personas/&lt;name&gt;/migrate-in.json</code>{" "}
          and run from terminal:
        </div>
        <pre
          style={{
            fontSize: 10.5,
            fontFamily: "DM Mono, Courier New, monospace",
            background: "rgba(0,0,0,0.06)",
            color: "var(--text)",
            padding: "8px 10px",
            borderRadius: 5,
            marginTop: 6,
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {`nell migrate --input ./migrate-in.json --persona <name>`}
        </pre>
        <div
          style={{
            marginTop: 6,
            fontSize: 10.5,
            color: "var(--text-mute)",
            lineHeight: 1.55,
            fontStyle: "italic",
          }}
        >
          A guided one-click importer for emergence-kit is on the
          roadmap; for now Path A is what most people choose.
        </div>
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
