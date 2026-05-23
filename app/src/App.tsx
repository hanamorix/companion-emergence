import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
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
import { fetchPersonaState, type PersonaState } from "./bridge";
import { NellAvatar } from "./components/NellAvatar";
import { ChatPanel } from "./components/ChatPanel";
import { LeftPanel } from "./components/LeftPanel";
import { useSoulFlash } from "./useSoulFlash";
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

  async function startPersona(persona: string) {
    setPhase({ kind: "starting-bridge", persona, error: null });
    try {
      await ensureBridgeRunning(persona);
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
    })();
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
  const soulFlashing = useSoulFlash(state);

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
        width: "100vw",
        height: "100vh",
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
          textbox without disturbing layout flow for the panels. */}
      <div style={{ transform: "translateY(50px)" }}>
        <NellAvatar
          state={state}
          persona={persona}
          isSpeaking={isSpeaking}
          soulFlashing={soulFlashing}
          reducedMotion={config.reduced_motion}
        />
      </div>
      <ChatPanel
        persona={persona}
        onSpeakingChange={setIsSpeaking}
        recovering={state?.recovering ?? false}
        feltTimeRecovered={state?.felt_time_recovered ?? false}
      />
    </div>
  );
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
