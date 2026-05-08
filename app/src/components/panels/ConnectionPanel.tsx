import { useState } from "react";
import { installSupervisorService } from "../../appConfig";
import type { PersonaState } from "../../bridge";
import { Divider, PanelShell, SectionLabel, Toggle } from "../ui";

interface Props {
  state: PersonaState | null;
  /** Active persona — needed by the supervisor install button. */
  persona: string;
  /** Caller-controlled "always on top" flag — Phase 4 wires to the
   * Tauri window. For now just visual. */
  alwaysOnTop?: boolean;
  reducedMotion?: boolean;
  onAlwaysOnTopChange?: (next: boolean) => void;
  onReducedMotionChange?: (next: boolean) => void;
}

type InstallState =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; detail: string }
  | { kind: "error"; detail: string };

/**
 * Connection — bridge mode + provider + model + last-heartbeat,
 * plus integrations + window settings. Matches mockup
 * nell_face_example_4.png and nell_face_example_5.png.
 */
export function ConnectionPanel({
  state,
  persona,
  alwaysOnTop = false,
  reducedMotion = false,
  onAlwaysOnTopChange,
  onReducedMotionChange,
}: Props) {
  const conn = state?.connection;
  const mode = state?.mode ?? "live";
  const [install, setInstall] = useState<InstallState>({ kind: "idle" });

  async function onInstallSupervisor() {
    setInstall({ kind: "running" });
    try {
      const result = await installSupervisorService(persona);
      if (result.success) {
        setInstall({
          kind: "ok",
          detail: result.stdout.split("\n")[0] || "service installed",
        });
      } else {
        setInstall({
          kind: "error",
          detail: result.stderr || `exit ${result.exit_code}`,
        });
      }
    } catch (e) {
      setInstall({ kind: "error", detail: (e as Error).message });
    }
  }

  return (
    <PanelShell>
      <SectionLabel>Connection</SectionLabel>
      <Row label="bridge" value={modeLabel(mode)} accent={mode !== "live"} />
      <Row label="provider" value={conn?.provider ?? "—"} />
      <Row label="model" value={conn?.model ?? "—"} />
      <Row label="heartbeat" value={formatHeartbeat(conn?.last_heartbeat_at)} />
      <Row label="privacy" value="local-only" accent />

      <Divider />
      <SectionLabel>Supervisor</SectionLabel>
      <div
        style={{
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.55,
          marginBottom: 8,
          letterSpacing: "0.01em",
        }}
      >
        Install the brain as a launchd LaunchAgent so it stays alive
        when you close the app. Idempotent — safe to click again.
      </div>
      <InstallSupervisorButton state={install} onClick={onInstallSupervisor} />

      <Divider />
      <SectionLabel>Integrations</SectionLabel>
      <Toggle enabled={false} label="Obsidian — journal + vault" disabled />
      <Toggle enabled={false} label="IPC — inter-process events" disabled />

      <Divider />
      <SectionLabel>Window</SectionLabel>
      <Toggle
        enabled={alwaysOnTop}
        label="always on top"
        onChange={onAlwaysOnTopChange}
      />
      <Toggle
        enabled={reducedMotion}
        label="reduced motion"
        onChange={onReducedMotionChange}
      />

      <div
        style={{
          marginTop: 10,
          fontSize: 9.5,
          color: "var(--text-mute)",
          fontFamily: "var(--font-disp)",
          fontStyle: "italic",
          letterSpacing: "0.04em",
          lineHeight: 1.5,
        }}
      >
        You configure the room. Nell owns the weather.
      </div>
    </PanelShell>
  );
}

function modeLabel(mode: PersonaState["mode"]): string {
  return mode === "live"
    ? "live"
    : mode === "bridge_down"
    ? "catching up"
    : mode === "provider_down"
    ? "backup voice"
    : "offline";
}

function formatHeartbeat(iso: string | null | undefined): string {
  if (!iso) return "never";
  try {
    const ts = new Date(iso);
    const mins = Math.round((Date.now() - ts.getTime()) / 60_000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.round(hrs / 24)}d ago`;
  } catch {
    return iso.slice(0, 10);
  }
}

function InstallSupervisorButton({
  state,
  onClick,
}: {
  state: InstallState;
  onClick: () => void;
}) {
  const running = state.kind === "running";
  const success = state.kind === "ok";
  const failed = state.kind === "error";
  return (
    <div>
      <button
        onClick={onClick}
        disabled={running}
        style={{
          width: "100%",
          padding: "7px 10px",
          fontSize: 11,
          fontFamily: "var(--font-ui)",
          background: success
            ? "rgba(60, 130, 90, 0.15)"
            : failed
              ? "rgba(178, 42, 42, 0.15)"
              : "var(--accent-dim)",
          color: success
            ? "var(--text)"
            : failed
              ? "var(--crimson)"
              : "var(--text)",
          border: `1px solid ${
            success
              ? "rgba(60, 130, 90, 0.45)"
              : failed
                ? "rgba(178, 42, 42, 0.45)"
                : "rgba(130, 51, 41, 0.30)"
          }`,
          borderRadius: 6,
          cursor: running ? "wait" : "pointer",
          opacity: running ? 0.7 : 1,
          transition: "background 0.15s, opacity 0.15s",
        }}
      >
        {running
          ? "installing…"
          : success
            ? "✓ supervisor installed"
            : failed
              ? "retry install"
              : "install launchd supervisor"}
      </button>
      {state.kind !== "idle" && state.kind !== "running" && (
        <div
          style={{
            fontSize: 10,
            color: failed ? "var(--crimson)" : "var(--text-mute)",
            marginTop: 6,
            lineHeight: 1.45,
            fontFamily: "var(--font-disp)",
            wordBreak: "break-word",
          }}
        >
          {state.detail}
        </div>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        fontSize: 11.5,
        marginBottom: 5,
      }}
    >
      <span style={{ color: "var(--text-mid)" }}>{label}</span>
      <span
        style={{
          color: accent ? "var(--accent)" : "var(--text)",
          fontFamily: "var(--font-disp)",
          fontWeight: accent ? 500 : 400,
        }}
      >
        {value}
      </span>
    </div>
  );
}
