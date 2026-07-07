import { useCallback, useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  brainLoginStatus,
  ensureBridgeRunning,
  listPersonas,
  nellbrainHomePath,
  readAppConfig,
  revealInFileManager,
  setAlwaysOnTop,
  writeAppConfig,
  type AppConfig,
  type PersonaSummary,
} from "./appConfig";

function dragWindow(e: React.MouseEvent) {
  if (e.button !== 0) return;
  void getCurrentWindow().startDragging();
}
import {
  approvePendingWrite,
  declinePendingWrite,
  fetchPersonaState,
  type PendingWrite,
  type PersonaState,
} from "./bridge";
import { ensureBridgeCurrent } from "./bridgeVersionCheck";
import { NellAvatar } from "./components/NellAvatar";
import { ChatPanel } from "./components/ChatPanel";
import { PendingWriteCard } from "./components/PendingWriteCard";
import { BrainLoginPrompt } from "./components/panels/BrainLoginPrompt";
import { LeftPanel } from "./components/LeftPanel";
import { useSoulFlash } from "./useSoulFlash";
import { useRestartBridge } from "./hooks/useRestartBridge";
import { Wizard } from "./wizard/Wizard";
import { PersonaPicker } from "./wizard/PersonaPicker";
import { errString } from "./lib/errString";

const STATE_POLL_MS = 5000;

type AppPhase =
  | { kind: "loading" }
  | { kind: "wizard" }
  | { kind: "picker"; personas: PersonaSummary[] }
  | { kind: "starting-bridge"; persona: string; error: string | null }
  | { kind: "ready"; persona: string };

/**
 * Companion Emergence top-level routing.
 *
 *   loading           — initial config read
 *   wizard            — first launch (no persona) or re-init triggered
 *   starting-bridge   — calling ensure_bridge_running for the persona
 *   ready             — main UI: avatar + panels + chat
 */
export default function App() {
  const [phase, setPhase] = useState<AppPhase>({ kind: "loading" });
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [versionMismatch, setVersionMismatch] = useState(false);

  async function startPersona(persona: string) {
    setPhase({ kind: "starting-bridge", persona, error: null });
    try {
      await ensureBridgeRunning(persona);
      const versionResult = await ensureBridgeCurrent(persona);
      setVersionMismatch(versionResult === "version_mismatch_unresolved");
      setPhase({ kind: "ready", persona });
    } catch (e) {
      setPhase({ kind: "starting-bridge", persona, error: errString(e) });
    }
  }

  // Boot: read config, decide first-launch vs ready, ensure bridge, etc.
  useEffect(() => {
    (async () => {
      const cfg = await readAppConfig();
      setConfig(cfg);
      if (cfg.selected_persona) {
        await startPersona(cfg.selected_persona);
        return;
      }
      const onDisk = await listPersonas();
      if (onDisk.length === 0) {
        setPhase({ kind: "wizard" });
        return;
      }
      if (onDisk.length === 1) {
        await writeAppConfig({ ...cfg, selected_persona: onDisk[0].name });
        await startPersona(onDisk[0].name);
        return;
      }
      setPhase({ kind: "picker", personas: onDisk });
    })().catch((e) => {
      console.error("[App] boot failed:", e);
    });
  }, []);

  if (phase.kind === "loading") return <BootScreen subtitle="Loading…" />;

  if (phase.kind === "picker") {
    return (
      <PersonaPicker
        personas={phase.personas}
        onPick={async (name) => {
          await writeAppConfig({ ...config!, selected_persona: name });
          setPhase({ kind: "loading" });
          await startPersona(name);
        }}
        onNew={() => setPhase({ kind: "wizard" })}
      />
    );
  }

  if (phase.kind === "wizard") {
    return (
      <Wizard
        onDone={async (persona) => {
          // Wizard already wrote app_config — but re-read to pick up
          // any defaults the install step may have set.
          const cfg = await readAppConfig();
          setConfig(cfg);
          await startPersona(persona);
        }}
      />
    );
  }

  if (phase.kind === "starting-bridge") {
    if (phase.error) {
      return (
        <BridgeErrorScreen
          persona={phase.persona}
          error={phase.error}
          onRetry={() => void startPersona(phase.persona)}
          onOpenAnyway={() => setPhase({ kind: "ready", persona: phase.persona })}
          onRunSetup={() => setPhase({ kind: "wizard" })}
        />
      );
    }
    return (
      <BootScreen
        subtitle={`Starting brain for ${phase.persona}…`}
      />
    );
  }

  return (
    <>
      <div className="titlebar-drag" onMouseDown={dragWindow} />
      {versionMismatch && (
        <div
          role="status"
          style={{
            position: "absolute",
            bottom: 8,
            left: "50%",
            transform: "translateX(-50%)",
            fontSize: 10,
            color: "var(--mauve)",
            background: "rgba(30,24,26,0.82)",
            padding: "3px 10px",
            borderRadius: 6,
            pointerEvents: "none",
            zIndex: 100,
            whiteSpace: "nowrap",
          }}
        >
          Companion's brain is running a different version — restart the app or reboot if things misbehave.
        </div>
      )}
      <Ready config={config!} setConfig={setConfig} persona={phase.persona} />
    </>
  );
}

function BridgeErrorScreen({
  persona,
  error,
  onRetry,
  onOpenAnyway,
  onRunSetup,
}: {
  persona: string;
  error: string;
  onRetry: () => void;
  onOpenAnyway: () => void;
  onRunSetup: () => void;
}) {
  const diagnostics = `persona=${persona}\nbridge_start_error=${error}`;
  const [logPath, setLogPath] = useState<string | null>(null);

  useEffect(() => {
    nellbrainHomePath().then((home) => {
      if (home) setLogPath(`${home}/launch-failures.log`);
    });
  }, []);

  async function copyDiagnostics() {
    try {
      await navigator.clipboard.writeText(diagnostics);
    } catch {
      // Clipboard is best-effort only; the visible diagnostics remain on screen.
    }
  }

  return (
    <>
      <div className="titlebar-drag" onMouseDown={dragWindow} />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 14,
          width: "100vw",
          height: "100vh",
          padding: 28,
          textAlign: "center",
        }}
      >
        <div style={{ fontFamily: "var(--font-disp)", color: "var(--linen)", fontSize: 18 }}>
          Brain startup needs attention
        </div>
        <div style={{ maxWidth: 420, color: "var(--mauve)", fontSize: 12, lineHeight: 1.6 }}>
          Companion Emergence could not start the bridge for <strong>{persona}</strong>.
          You can retry, open the app in degraded mode, or re-run setup.
        </div>
        <pre
          style={{
            maxWidth: 460,
            maxHeight: 120,
            overflow: "auto",
            whiteSpace: "pre-wrap",
            textAlign: "left",
            padding: 10,
            borderRadius: 8,
            background: "rgba(234,222,218,0.08)",
            border: "1px solid rgba(234,222,218,0.18)",
            color: "var(--linen)",
            fontSize: 11,
          }}
        >
          {diagnostics}
        </pre>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
          <BootButton onClick={onRetry}>Retry</BootButton>
          <BootButton onClick={onOpenAnyway}>Open degraded</BootButton>
          <BootButton onClick={onRunSetup}>Re-run setup</BootButton>
          <BootButton onClick={() => void copyDiagnostics()}>Copy diagnostics</BootButton>
        </div>
        {logPath && (
          <div style={{ marginTop: 16, fontSize: 11, color: "var(--mauve)", textAlign: "left", maxWidth: 460 }}>
            <div style={{ marginBottom: 6 }}>
              More detail: <code style={{ fontSize: 10.5, wordBreak: "break-all" }}>{logPath}</code>
            </div>
            <BootButton onClick={() => void revealInFileManager(logPath)}>
              Open log folder
            </BootButton>
          </div>
        )}
      </div>
    </>
  );
}

function BootButton(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      style={{
        padding: "7px 10px",
        borderRadius: 7,
        border: "1px solid rgba(234,222,218,0.25)",
        background: "rgba(234,222,218,0.12)",
        color: "var(--linen)",
        fontSize: 11,
        cursor: "pointer",
      }}
    />
  );
}

function BootScreen({ subtitle }: { subtitle: string }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 16,
        width: "100vw",
        height: "100vh",
      }}
    >
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: "50%",
          border: "2px solid rgba(234,222,218,0.2)",
          borderTopColor: "var(--linen)",
          animation: "spin 0.9s linear infinite",
        }}
      />
      <div
        style={{
          fontSize: 11,
          color: "var(--mauve)",
          fontFamily: "var(--font-disp)",
          letterSpacing: "0.06em",
          fontStyle: "italic",
        }}
      >
        {subtitle}
      </div>
    </div>
  );
}

interface ReadyProps {
  config: AppConfig;
  setConfig: (c: AppConfig) => void;
  persona: string;
}

function Ready({ config, setConfig, persona }: ReadyProps) {
  const [state, setState] = useState<PersonaState | null>(null);
  const [stateError, setStateError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  // Ids of pending writes the user has just acted on — optimistically hidden
  // until the next /persona/state poll drops them server-side. Also disables
  // that card's buttons so a double-click can't double-approve.
  const [resolvingWrites, setResolvingWrites] = useState<Set<string>>(new Set());
  const soulFlashing = useSoulFlash(state);

  // Brain clean-login banner — an offer, never a gate. LOAD-BEARING
  // INVARIANT: chat renders unconditionally regardless of these values.
  // null = unknown/still-checking (no banner); session-only dismissal
  // means the offer is re-shown next app open while still unauthorized.
  const [brainAuthorized, setBrainAuthorized] = useState<boolean | null>(null);
  const [brainPromptDismissed, setBrainPromptDismissed] = useState(false);
  const restartBridge = useRestartBridge(persona, state?.mode ?? "live");

  useEffect(() => {
    let cancelled = false;
    brainLoginStatus()
      .then((res) => {
        if (!cancelled) setBrainAuthorized(res.authorized);
      })
      .catch(() => {
        // Swallow — absence of a status is treated as "no banner", never
        // a crash and never a block on chat.
        if (!cancelled) setBrainAuthorized(true);
      });
    return () => {
      cancelled = true;
    };
  }, [persona]);

  const refetchState = useCallback(async () => {
    try {
      const s = await fetchPersonaState(persona);
      setState(s);
      setStateError(null);
    } catch (e) {
      setStateError(errString(e));
    }
  }, [persona]);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const s = await fetchPersonaState(persona);
        if (!cancelled) {
          setState(s);
          setStateError(null);
        }
      } catch (e) {
        if (!cancelled) setStateError(errString(e));
      }
    }
    tick();
    const id = setInterval(tick, STATE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [persona]);

  // Approve/decline a proposed file write. Mirror the reach-out card resolve:
  // mark busy, call the endpoint, optimistic-remove + refetch. On failure,
  // un-busy so the user can retry (the card reappears on the next poll if the
  // write is still pending server-side).
  async function resolveWrite(
    id: string,
    action: (persona: string, id: string) => Promise<unknown>,
  ) {
    setResolvingWrites((prev) => new Set(prev).add(id));
    try {
      await action(persona, id);
      await refetchState();
    } catch {
      setResolvingWrites((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  const pendingWrites: PendingWrite[] = (state?.pending_writes ?? []).filter(
    (w) => !resolvingWrites.has(w.id),
  );

  useEffect(() => {
    document.documentElement.dataset.reducedMotion = config.reduced_motion ? "true" : "false";
  }, [config.reduced_motion]);

  // Apply always-on-top to the actual Tauri window — both on mount
  // (for the saved value) and whenever the user toggles it.
  useEffect(() => {
    void setAlwaysOnTop(config.always_on_top);
  }, [config.always_on_top]);

  function updateConfig(patch: Partial<AppConfig>) {
    const next = { ...config, ...patch };
    setConfig(next);
    void writeAppConfig(next);
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 28,
        width: "100%",
        height: "100%",
      }}
    >
      <GlobalStatusDot mode={state?.mode ?? "live"} stateError={stateError} />
      <LeftPanel
        state={state}
        persona={persona}
        stateError={stateError}
        alwaysOnTop={config.always_on_top}
        reducedMotion={config.reduced_motion}
        onAlwaysOnTopChange={(v) => updateConfig({ always_on_top: v })}
        onReducedMotionChange={(v) => updateConfig({ reduced_motion: v })}
      />
      {/* The flex row centers everyone vertically, but the chat panel
          is taller (380) than the avatar (280), so avatar's bottom
          floats ~50px above the chat input. translateY drops the
          avatar so her silhouette ends at the same line as the
          textbox without disturbing layout flow for the panels. The
          identity block (name/status/heartbeat) sits under the avatar
          in its own column. */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{ transform: "translateY(50px)" }}>
          <NellAvatar
            state={state}
            persona={persona}
            isSpeaking={isSpeaking}
            soulFlashing={soulFlashing}
            reducedMotion={config.reduced_motion}
          />
        </div>
        <PresenceIdentity persona={persona} state={state} isSpeaking={isSpeaking} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {pendingWrites.length > 0 && (
          <div
            className="pending-write-list"
            data-testid="pending-write-list"
            role="list"
            aria-label="Proposed file writes"
          >
            {pendingWrites.map((w) => (
              <PendingWriteCard
                key={w.id}
                write={w}
                busy={resolvingWrites.has(w.id)}
                onApprove={(id) => void resolveWrite(id, approvePendingWrite)}
                onDecline={(id) => void resolveWrite(id, declinePendingWrite)}
              />
            ))}
          </div>
        )}
        {brainAuthorized === false && !brainPromptDismissed && (
          <div
            data-testid="brain-login-banner"
            style={{
              border: "1px solid var(--panel-border, rgba(234,222,218,0.18))",
              borderRadius: 8,
              padding: 10,
              background: "var(--panel-bg)",
            }}
          >
            <BrainLoginPrompt
              onAuthorized={() => {
                setBrainAuthorized(true);
                restartBridge.restart();
              }}
              onDismiss={() => setBrainPromptDismissed(true)}
            />
          </div>
        )}
        <ChatPanel
          persona={persona}
          onSpeakingChange={setIsSpeaking}
          recovering={state?.recovering ?? false}
          feltTimeRecovered={state?.felt_time_recovered ?? false}
          mode={stateError ? "bridge_down" : (state?.mode ?? "live")}
        />
      </div>
    </div>
  );
}

/**
 * PresenceIdentity — the identity block under the avatar: a live dot,
 * the persona name, a humanized status line derived from her top-2
 * emotions (swaps to "thinking…" while a reply is in flight), and a
 * heartbeat chip. Purely presentational — reads only from PersonaState
 * already polled by Ready.
 */
function PresenceIdentity({
  persona,
  state,
  isSpeaking,
}: {
  persona: string;
  state: PersonaState | null;
  isSpeaking: boolean;
}) {
  const statusLine = isSpeaking ? "thinking…" : humanizeEmotionStatus(state?.emotions ?? null);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 18, marginTop: 18 }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            aria-hidden="true"
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#5fbe8b",
              boxShadow: "0 0 8px rgba(95,190,139,0.7)",
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 19, fontWeight: 700, letterSpacing: "-0.01em", color: "var(--text)" }}>
            {capitalize(persona)}
          </span>
        </div>
        {statusLine && (
          <div style={{ fontSize: 12, color: "var(--text-mute)" }}>{statusLine}</div>
        )}
      </div>
      <div
        style={{
          fontSize: 11,
          fontWeight: 500,
          padding: "5px 13px",
          borderRadius: 999,
          color: "var(--text-mid)",
          background: "var(--panel)",
          border: "1px solid var(--hairline)",
        }}
      >
        ♥ {nextHeartbeatLabel(state?.connection?.last_heartbeat_at)}
      </div>
    </div>
  );
}

/** Persisted supervisor cadence — see brain/bridge/supervisor.py
 *  heartbeat_interval_s default. Not exposed on PersonaState today, so
 *  this mirrors the known server-side constant to derive a "next
 *  heartbeat in Xm" display from the same last_heartbeat_at field
 *  ConnectionPanel's heartbeat row already reads (that row shows time
 *  SINCE the last heartbeat; this derives time UNTIL the next one from
 *  the same timestamp + the fixed interval). */
const HEARTBEAT_INTERVAL_MINUTES = 15;

function nextHeartbeatLabel(iso: string | null | undefined): string {
  if (!iso) return "next heartbeat soon";
  try {
    const ts = new Date(iso).getTime();
    const elapsedMin = (Date.now() - ts) / 60_000;
    const remaining = Math.max(0, Math.ceil(HEARTBEAT_INTERVAL_MINUTES - elapsedMin));
    if (remaining <= 0) return "next heartbeat any moment";
    return `next heartbeat in ${remaining}m`;
  } catch {
    return "next heartbeat soon";
  }
}

/** Humanizes the top-2 emotion channels into a short status phrase, e.g.
 *  "creative hunger climbing · rest need underneath". Snake_case channel
 *  names become space-separated lowercase phrases; exact adjective framing
 *  is a visual/creative choice, kept short. */
function humanizeEmotionStatus(emotions: Record<string, number> | null): string | null {
  if (!emotions) return null;
  const top = Object.entries(emotions)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 2)
    .map(([name]) => name.replace(/_/g, " ").toLowerCase());
  if (top.length === 0) return null;
  if (top.length === 1) return `${top[0]} climbing`;
  return `${top[0]} climbing · ${top[1]} underneath`;
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

/**
 * GlobalStatusDot — always-visible degraded-state indicator.
 *
 * Renders nothing in the healthy case so the UI stays clean. Surfaces
 * a small floating pill at the top-right of the app whenever:
 *
 *  - state polling has failed (`stateError`), or
 *  - bridge mode is anything but "live".
 *
 * Crimson dot for hard errors (state-poll failure, bridge offline /
 * down), amber for soft warnings (provider down). Tooltip describes
 * the issue for screen readers + hover. The full detail still lives
 * in the Connection panel's StatusBanner; this exists so the user
 * doesn't have to open a panel to know something's wrong.
 */
function GlobalStatusDot({
  mode,
  stateError,
}: {
  mode: PersonaState["mode"];
  stateError: string | null;
}) {
  let kind: "error" | "warn" | null = null;
  let title = "";
  if (stateError) {
    kind = "error";
    title = `State poll failed: ${stateError}`;
  } else if (mode === "bridge_down" || mode === "offline") {
    kind = "error";
    title = mode === "offline" ? "Brain offline." : "Bridge offline.";
  } else if (mode === "provider_down") {
    kind = "warn";
    title = "LLM provider unreachable — backup voice in use.";
  }
  if (kind === null) return null;

  const palette =
    kind === "error"
      ? { bg: "rgba(178, 42, 42, 0.95)", ring: "rgba(178, 42, 42, 0.35)" }
      : { bg: "rgba(216, 154, 88, 0.95)", ring: "rgba(216, 154, 88, 0.35)" };

  return (
    <div
      role="status"
      aria-live="polite"
      title={title}
      style={{
        position: "absolute",
        top: 8,
        right: 12,
        width: 12,
        height: 12,
        borderRadius: "50%",
        background: palette.bg,
        boxShadow: `0 0 0 4px ${palette.ring}`,
        zIndex: 50,
        animation: "pulse 1.6s ease-in-out infinite",
      }}
    />
  );
}
