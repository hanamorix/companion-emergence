import { useEffect, type ReactNode } from "react";
import {
  Divider,
  FieldLabel,
  OptionCard,
  SectionLabel,
  WButton,
  WInput,
  WizardShell,
} from "../components";
import {
  runPreflightExistingCE,
  type ExistingCePreflight,
  type MigrateSource,
} from "../../appConfig";
import { errString } from "../../lib/errString";

interface Props {
  step: number;
  totalSteps: number;
  path: string;
  onPathChange: (p: string) => void;
  source: MigrateSource;
  onSourceChange: (s: MigrateSource) => void;
  preflight: ExistingCePreflight | null;
  onPreflightChange: (p: ExistingCePreflight | null) => void;
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
  preflight,
  onPreflightChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  // For companion-emergence, the preflight must be ok before we can proceed.
  // For the other two sources, a non-empty path is enough.
  const valid = source === "companion-emergence"
    ? (preflight?.ok ?? false)
    : path.trim().length > 0;

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
        <OptionCard
          selected={source === "companion-emergence"}
          onClick={() => onSourceChange("companion-emergence")}
          title="An existing companion-emergence install"
          description="You've already used Companion Emergence — upgrading from an older version, or moving to a new machine. Point at the persona folder."
        />
      </div>

      <Divider />

      {source === "nellbrain" && <NellBrainPath path={path} onPathChange={onPathChange} />}
      {source === "emergence-kit" && <EmergenceKitGuide path={path} onPathChange={onPathChange} />}
      {source === "companion-emergence" && (
        <ExistingCEPath
          path={path}
          onPathChange={onPathChange}
          preflight={preflight}
          onPreflightChange={onPreflightChange}
        />
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

function ExistingCEPath({
  path,
  onPathChange,
  preflight,
  onPreflightChange,
}: {
  path: string;
  onPathChange: (p: string) => void;
  preflight: ExistingCePreflight | null;
  onPreflightChange: (p: ExistingCePreflight | null) => void;
}) {
  // Debounced preflight — fires 500 ms after the path stops changing.
  useEffect(() => {
    if (!path.trim()) {
      onPreflightChange(null);
      return;
    }
    const t = setTimeout(async () => {
      try {
        onPreflightChange(await runPreflightExistingCE(path));
      } catch (e) {
        onPreflightChange({
          ok: false,
          persona_name: null,
          imported_user_name: null,
          imported_voice_template: null,
          memory_count: null,
          crystallization_count: null,
          hebbian_edge_count: null,
          source_size_bytes: 0,
          errors: [{ code: "preflight_failed", message: errString(e) }],
          warnings: [],
        });
      }
    }, 500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);

  // If preflight returned a "pointed_at_parent" error, the detail may contain
  // subdirectory names that are valid persona dirs.
  const parentError = preflight?.errors.find((e) => e.code === "pointed_at_parent");
  const suggestedSubdirs =
    (parentError?.detail as { suggested_subdirs?: string[] } | undefined)?.suggested_subdirs ?? [];

  return (
    <>
      <SectionLabel>Where is your existing Kindled?</SectionLabel>
      <WInput
        value={path}
        onChange={onPathChange}
        placeholder={defaultExistingCePlaceholder()}
        mono
      />

      {/* Preflight error state — red callout */}
      {preflight && !preflight.ok && (
        <div
          style={{
            marginTop: 8,
            padding: "10px 12px",
            borderRadius: 7,
            background: "rgba(160, 30, 30, 0.07)",
            border: "1px solid rgba(160, 30, 30, 0.35)",
          }}
        >
          {preflight.errors.map((e) => (
            <div
              key={e.code}
              style={{ fontSize: 11, color: "var(--crimson)", lineHeight: 1.6 }}
            >
              {e.message}
            </div>
          ))}
          {suggestedSubdirs.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div
                style={{
                  fontSize: 10.5,
                  color: "var(--text-mute)",
                  marginBottom: 4,
                  fontFamily: "var(--font-disp)",
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                }}
              >
                Did you mean:
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {suggestedSubdirs.map((name) => (
                  <button
                    key={name}
                    onClick={() =>
                      onPathChange(`${path.replace(/\/+$/, "")}/${name}`)
                    }
                    style={{
                      fontSize: 10.5,
                      fontFamily: "DM Mono, Courier New, monospace",
                      background: "rgba(130,51,41,0.10)",
                      border: "1px solid var(--accent)",
                      borderRadius: 5,
                      padding: "3px 8px",
                      cursor: "pointer",
                      color: "var(--accent)",
                    }}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Preflight success state — green callout */}
      {preflight?.ok && (
        <div
          style={{
            marginTop: 8,
            padding: "10px 12px",
            borderRadius: 7,
            background: "rgba(40, 120, 60, 0.07)",
            border: "1px solid rgba(40, 120, 60, 0.35)",
          }}
        >
          <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 6, fontWeight: 500 }}>
            Found <strong>{preflight.persona_name}</strong>
          </div>
          <div
            style={{
              fontSize: 10.5,
              color: "var(--text-mid)",
              lineHeight: 1.7,
            }}
          >
            {preflight.memory_count != null && (
              <div>{preflight.memory_count} memories</div>
            )}
            {preflight.crystallization_count != null && (
              <div>{preflight.crystallization_count} crystallizations</div>
            )}
            {preflight.hebbian_edge_count != null && (
              <div>{preflight.hebbian_edge_count} Hebbian edges</div>
            )}
            {preflight.imported_user_name && (
              <div>user: {preflight.imported_user_name}</div>
            )}
            {preflight.imported_voice_template && (
              <div>voice: {preflight.imported_voice_template}</div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function defaultExistingCePlaceholder(): string {
  const ua = navigator.userAgent;
  if (/win/i.test(ua))
    return "%LOCALAPPDATA%\\hanamorix\\companion-emergence\\personas\\<name>";
  if (/mac/i.test(ua))
    return "~/Library/Application Support/companion-emergence/personas/<name>";
  return "~/.local/share/companion-emergence/personas/<name>";
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
