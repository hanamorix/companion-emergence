import { useState } from "react";
import type { PersonaState } from "../bridge";
import { InnerWeatherPanel } from "./panels/InnerWeatherPanel";
import { BodyPanel } from "./panels/BodyPanel";
import { InteriorPanel } from "./panels/InteriorPanel";
import { SoulPanel } from "./panels/SoulPanel";
import { ConnectionPanel } from "./panels/ConnectionPanel";

type Tab = "weather" | "body" | "interior" | "soul" | "connection";

interface Props {
  state: PersonaState | null;
  persona: string;
  stateError?: string | null;
  alwaysOnTop: boolean;
  reducedMotion: boolean;
  onAlwaysOnTopChange: (next: boolean) => void;
  onReducedMotionChange: (next: boolean) => void;
}

const TABS: Array<{ id: Tab; label: string; icon: string }> = [
  { id: "weather", label: "Inner Weather", icon: "◐" },
  { id: "body", label: "Body", icon: "○" },
  { id: "interior", label: "Recent Interior", icon: "✦" },
  { id: "soul", label: "Soul", icon: "❀" },
  { id: "connection", label: "Connection", icon: "≡" },
];

export function LeftPanel({
  state,
  persona,
  stateError = null,
  alwaysOnTop,
  reducedMotion,
  onAlwaysOnTopChange,
  onReducedMotionChange,
}: Props) {
  const [tab, setTab] = useState<Tab>("weather");
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
      {renderPanel(tab, state, {
        persona,
        stateError,
        alwaysOnTop,
        reducedMotion,
        onAlwaysOnTopChange,
        onReducedMotionChange,
      })}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 6,
          padding: 6,
          background: "var(--panel-bg)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          boxShadow:
            "0 1px 2px rgba(42,31,31,0.06), inset 0 0 0 1px rgba(130,51,41,0.08)",
        }}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            title={t.label}
            aria-label={t.label}
            style={{
              width: 26,
              height: 26,
              borderRadius: 6,
              background: tab === t.id ? "var(--accent)" : "transparent",
              color: tab === t.id ? "var(--linen)" : "var(--text-mid)",
              border:
                tab === t.id
                  ? "1px solid var(--accent)"
                  : "1px solid rgba(130,51,41,0.18)",
              fontSize: 13,
              transition: "all 0.18s ease",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {t.icon}
          </button>
        ))}
      </div>
    </div>
  );
}

interface PanelOpts {
  persona: string;
  stateError: string | null;
  alwaysOnTop: boolean;
  reducedMotion: boolean;
  onAlwaysOnTopChange: (next: boolean) => void;
  onReducedMotionChange: (next: boolean) => void;
}

function renderPanel(tab: Tab, state: PersonaState | null, opts: PanelOpts) {
  switch (tab) {
    case "weather":
      return <InnerWeatherPanel state={state} />;
    case "body":
      return <BodyPanel state={state} />;
    case "interior":
      return <InteriorPanel state={state} />;
    case "soul":
      return <SoulPanel state={state} />;
    case "connection":
      return (
        <ConnectionPanel
          state={state}
          persona={opts.persona}
          stateError={opts.stateError}
          alwaysOnTop={opts.alwaysOnTop}
          reducedMotion={opts.reducedMotion}
          onAlwaysOnTopChange={opts.onAlwaysOnTopChange}
          onReducedMotionChange={opts.onReducedMotionChange}
        />
      );
  }
}
