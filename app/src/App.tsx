import { useEffect, useState } from "react";
import { fetchPersonaState, type PersonaState } from "./bridge";
import { NellAvatar } from "./components/NellAvatar";
import { ChatPanel } from "./components/ChatPanel";
import { LeftPanel } from "./components/LeftPanel";

const STATE_POLL_MS = 5000;

/**
 * NellFace shell — three columns: LeftPanel / NellAvatar / ChatPanel.
 *
 * Left panel rotates through Inner Weather / Body / Recent Interior /
 * Soul / Connection via the icon column. All five read the same
 * /persona/state poll.
 */
export default function App() {
  const [state, setState] = useState<PersonaState | null>(null);
  const [stateError, setStateError] = useState<string | null>(null);
  const [alwaysOnTop, setAlwaysOnTop] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const s = await fetchPersonaState();
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
  }, []);

  // User-controlled reduced motion — sets a data attribute on <html>
  // that styles.css picks up. Phase 4 will sync this with PersonaConfig.
  useEffect(() => {
    document.documentElement.dataset.reducedMotion = reducedMotion ? "true" : "false";
  }, [reducedMotion]);

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
        alwaysOnTop={alwaysOnTop}
        reducedMotion={reducedMotion}
        onAlwaysOnTopChange={setAlwaysOnTop}
        onReducedMotionChange={setReducedMotion}
      />

      <NellAvatar state={state} />

      <ChatPanel />

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
