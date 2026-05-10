import { useState } from "react";
import { installNellCliSymlink, installSupervisorService } from "../../appConfig";
import type { PersonaState } from "../../bridge";
import { Divider, PanelShell, SectionLabel, Toggle } from "../ui";

interface Props {
  state: PersonaState | null;
  /** Active persona — needed by the supervisor install button. */
  persona: string;
  /** Last `/state` poll error, if any — surfaced to the user in the
   * Status section so silent failures don't go unnoticed. */
  stateError?: string | null;
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
 * Connection — bridge mode, provider, model, last-heartbeat, status
 * warnings, supervisor install, and window settings.
 */
export function ConnectionPanel({
  state,
  persona,
  stateError = null,
  alwaysOnTop = false,
  reducedMotion = false,
  onAlwaysOnTopChange,
  onReducedMotionChange,
}: Props) {
  const conn = state?.connection;
  const mode = state?.mode ?? "live";
  const [install, setInstall] = useState<InstallState>({ kind: "idle" });
  const [cliInstall, setCliInstall] = useState<InstallState>({ kind: "idle" });

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

  async function onInstallCli() {
    setCliInstall({ kind: "running" });
    try {
      const result = await installNellCliSymlink();
      if (result.success) {
        setCliInstall({
          kind: "ok",
          // Show full stdout (link + PATH hint, ~2 lines).
          detail: result.stdout.trim() || "nell installed",
        });
      } else {
        setCliInstall({
          kind: "error",
          detail: result.stderr || `exit ${result.exit_code}`,
        });
      }
    } catch (e) {
      setCliInstall({ kind: "error", detail: (e as Error).message });
    }
  }

  return (
    <PanelShell>
      <SectionLabel>Connection</SectionLabel>
      <Row label="Bridge" value={modeLabel(mode)} accent={mode !== "live"} />
      <Row label="Provider" value={conn?.provider ?? "—"} />
      <Row label="Model" value={conn?.model ?? "—"} />
      <Row label="Heartbeat" value={formatHeartbeat(conn?.last_heartbeat_at)} />
      <Row label="Privacy" value="Local-only" accent />

      <StatusBanner mode={mode} stateError={stateError} />

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
      <InstallActionButton
        state={install}
        onClick={onInstallSupervisor}
        idleLabel="install launchd supervisor"
        runningLabel="installing…"
        successLabel="✓ supervisor installed"
        errorLabel="retry install"
      />

      <Divider />
      <SectionLabel>Terminal</SectionLabel>
      <div
        style={{
          fontSize: 10.5,
          color: "var(--text-mute)",
          lineHeight: 1.55,
          marginBottom: 8,
          letterSpacing: "0.01em",
        }}
      >
        Add a <code>nell</code> shortcut to ~/.local/bin so you can run
        commands from Terminal. Idempotent — safe to click again.
      </div>
      <InstallActionButton
        state={cliInstall}
        onClick={onInstallCli}
        idleLabel="install nell to ~/.local/bin"
        runningLabel="installing…"
        successLabel="✓ nell installed"
        errorLabel="retry install"
      />

      <Divider />
      <SectionLabel>Window</SectionLabel>
      <Toggle
        enabled={alwaysOnTop}
        label="Always on top"
        onChange={onAlwaysOnTopChange}
      />
      <Toggle
        enabled={reducedMotion}
        label="Reduced motion"
        onChange={onReducedMotionChange}
      />
    </PanelShell>
  );
}

/**
 * StatusBanner — surfaces failures and degraded modes inside the
 * settings menu so silent issues don't go unnoticed. Renders nothing
 * when everything is live + no error; otherwise shows a coloured
 * banner with a one-line headline + the underlying detail.
 */
function StatusBanner({
  mode,
  stateError,
}: {
  mode: PersonaState["mode"];
  stateError: string | null;
}) {
  // Pick the worst signal — bridge_down is more critical than provider_down.
  let kind: "warning" | "error" | null = null;
  let headline = "";
  let detail = "";
  if (stateError) {
    kind = "error";
    headline = "State poll failed.";
    detail = stateError;
  } else if (mode === "bridge_down") {
    kind = "error";
    headline = "Bridge offline.";
    detail =
      "The brain isn't reachable. Try installing the launchd supervisor below, or run `nell service status` from terminal.";
  } else if (mode === "offline") {
    kind = "error";
    headline = "Offline.";
    detail = "No bridge or provider available. Chat is disabled until the brain comes back.";
  } else if (mode === "provider_down") {
    kind = "warning";
    headline = "LLM provider unreachable.";
    detail =
      "Replies will fall back to a local backup voice. Check that `claude` is on PATH for the launchd agent.";
  }

  if (kind === null) return null;

  const palette =
    kind === "error"
      ? {
          bg: "rgba(178, 42, 42, 0.10)",
          border: "rgba(178, 42, 42, 0.40)",
          headline: "var(--crimson)",
        }
      : {
          bg: "rgba(216, 154, 88, 0.14)",
          border: "rgba(216, 154, 88, 0.45)",
          headline: "#a07434",
        };

  return (
    <div
      role="alert"
      style={{
        marginTop: 12,
        padding: "8px 10px",
        borderRadius: 6,
        background: palette.bg,
        border: `1px solid ${palette.border}`,
      }}
    >
      <div
        style={{
          fontSize: 11.5,
          fontWeight: 500,
          color: palette.headline,
          marginBottom: 3,
        }}
      >
        {headline}
      </div>
      <div
        style={{
          fontSize: 10.5,
          color: "var(--text-mid)",
          lineHeight: 1.5,
          wordBreak: "break-word",
        }}
      >
        {detail}
      </div>
    </div>
  );
}

function modeLabel(mode: PersonaState["mode"]): string {
  return mode === "live"
    ? "Live"
    : mode === "bridge_down"
    ? "Catching up"
    : mode === "provider_down"
    ? "Backup voice"
    : "Offline";
}

function formatHeartbeat(iso: string | null | undefined): string {
  if (!iso) return "Never";
  try {
    const ts = new Date(iso);
    const mins = Math.round((Date.now() - ts.getTime()) / 60_000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.round(hrs / 24)}d ago`;
  } catch {
    return iso.slice(0, 10);
  }
}

function InstallActionButton({
  state,
  onClick,
  idleLabel,
  runningLabel,
  successLabel,
  errorLabel,
}: {
  state: InstallState;
  onClick: () => void;
  idleLabel: string;
  runningLabel: string;
  successLabel: string;
  errorLabel: string;
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
        {running ? runningLabel : success ? successLabel : failed ? errorLabel : idleLabel}
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
            whiteSpace: "pre-wrap",
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
