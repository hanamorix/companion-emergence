import { useEffect, useState } from "react";
import {
  ensureBridgeRunning,
  readAppConfig,
  setAlwaysOnTop,
  writeAppConfig,
  type AppConfig,
} from "./appConfig";
import { fetchPersonaState, type PersonaState } from "./bridge";
import { NellAvatar } from "./components/NellAvatar";
import { ChatPanel } from "./components/ChatPanel";
import { LeftPanel } from "./components/LeftPanel";
import { useSoulFlash } from "./useSoulFlash";
import { Wizard } from "./wizard/Wizard";

const STATE_POLL_MS = 5000;

type AppPhase =
  | { kind: "loading" }
  | { kind: "wizard" }
  | { kind: "starting-bridge"; persona: string; error: string | null }
  | { kind: "ready"; persona: string };

/**
 * NellFace top-level routing.
 *
 *   loading           — initial config read
 *   wizard            — first launch (no persona) or re-init triggered
 *   starting-bridge   — calling ensure_bridge_running for the persona
 *   ready             — main NellFace UI: avatar + panels + chat
 */
export default function App() {
  const [phase, setPhase] = useState<AppPhase>({ kind: "loading" });
  const [config, setConfig] = useState<AppConfig | null>(null);

  // Boot: read config, decide first-launch vs ready, ensure bridge, etc.
  useEffect(() => {
    (async () => {
      const cfg = await readAppConfig();
      setConfig(cfg);
      if (!cfg.selected_persona) {
        setPhase({ kind: "wizard" });
        return;
      }
      setPhase({ kind: "starting-bridge", persona: cfg.selected_persona, error: null });
      try {
        await ensureBridgeRunning(cfg.selected_persona);
        setPhase({ kind: "ready", persona: cfg.selected_persona });
      } catch (e) {
        setPhase({
          kind: "starting-bridge",
          persona: cfg.selected_persona,
          error: (e as Error).message,
        });
      }
    })();
  }, []);

  if (phase.kind === "loading") return <BootScreen subtitle="loading…" />;

  if (phase.kind === "wizard") {
    return (
      <Wizard
        onDone={async (persona) => {
          // Wizard already wrote app_config — but re-read to pick up
          // any defaults the install step may have set.
          const cfg = await readAppConfig();
          setConfig(cfg);
          setPhase({ kind: "starting-bridge", persona, error: null });
          try {
            await ensureBridgeRunning(persona);
            setPhase({ kind: "ready", persona });
          } catch (e) {
            setPhase({ kind: "starting-bridge", persona, error: (e as Error).message });
          }
        }}
      />
    );
  }

  if (phase.kind === "starting-bridge") {
    return (
      <BootScreen
        subtitle={
          phase.error
            ? `bridge failed to start: ${phase.error}`
            : `starting brain for ${phase.persona}…`
        }
      />
    );
  }

  return <Ready config={config!} setConfig={setConfig} persona={phase.persona} />;
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
        if (!cancelled) setStateError((e as Error).message);
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
      <LeftPanel
        state={state}
        alwaysOnTop={config.always_on_top}
        reducedMotion={config.reduced_motion}
        onAlwaysOnTopChange={(v) => updateConfig({ always_on_top: v })}
        onReducedMotionChange={(v) => updateConfig({ reduced_motion: v })}
      />
      <NellAvatar
        state={state}
        isSpeaking={isSpeaking}
        soulFlashing={soulFlashing}
        reducedMotion={config.reduced_motion}
      />
      <ChatPanel persona={persona} onSpeakingChange={setIsSpeaking} />
      {stateError && (
        <div
          style={{
            position: "absolute",
            bottom: 8,
            left: 8,
            fontSize: 10,
            color: "var(--crimson)",
            fontFamily: "var(--font-disp)",
            opacity: 0.7,
          }}
        >
          state: {stateError}
        </div>
      )}
    </div>
  );
}
