import { useEffect, useState } from "react";
import { fetchPersonaState, type PersonaState } from "./bridge";
import { NellAvatar } from "./components/NellAvatar";
import { ChatPanel } from "./components/ChatPanel";

const STATE_POLL_MS = 5000;

/**
 * NellFace shell — three columns: TBD-left-panel / NellAvatar / ChatPanel.
 *
 * Phase 2 (this commit) ships avatar + chat working end-to-end against
 * the bridge. Phase 3 fills in the left panels (Inner Weather, Body,
 * Recent Interior, Soul, Connection) reading the same persona state.
 */
export default function App() {
  const [state, setState] = useState<PersonaState | null>(null);
  const [stateError, setStateError] = useState<string | null>(null);

  // Poll /persona/state every 5s. Switching to SSE/WS later when the
  // backend exposes a streaming surface.
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

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 36,
        width: "100vw",
        height: "100vh",
      }}
    >
      {/* Left column — placeholder until Phase 3 panels land */}
      <div
        style={{
          width: 220,
          minHeight: 280,
          opacity: 0.3,
          fontSize: 10,
          color: "var(--mauve)",
          fontFamily: "var(--font-disp)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          padding: 8,
        }}
      >
        {stateError ? `state: ${stateError}` : "panels — phase 3"}
      </div>

      {/* Avatar */}
      <NellAvatar state={state} />

      {/* Chat */}
      <ChatPanel />
    </div>
  );
}
