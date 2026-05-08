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
  // null = no panel open, just the icon rail. Click an icon to open
  // its panel; click the same icon again to close it. Lets the user
  // hide the cards when they want a clean view of the avatar + chat.
  const [tab, setTab] = useState<Tab | null>(null);
  // The icon rail stays in normal flow (its vertical extent defines
  // LeftPanel's box, which the parent flex centers). The panel floats
  // out to its left via absolute positioning when open, so toggling
  // a panel never changes LeftPanel's width — avatar + chat stay put.
  return (
    <div style={{ position: "relative" }}>
      {tab !== null && (
        <div
          style={{
            position: "absolute",
            top: 0,
            right: "calc(100% + 8px)",
          }}
        >
          {renderPanel(tab, state, {
            persona,
            stateError,
            alwaysOnTop,
            reducedMotion,
            onAlwaysOnTopChange,
            onReducedMotionChange,
          })}
        </div>
      )}
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
        {TABS.map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setTab((prev) => (prev === t.id ? null : t.id))}
              title={`${t.label}${active ? " (click to close)" : ""}`}
              aria-label={t.label}
              aria-pressed={active}
              style={{
                width: 26,
                height: 26,
                borderRadius: 6,
                background: active ? "var(--accent)" : "transparent",
                color: active ? "var(--linen)" : "var(--text-mid)",
                border: active
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
          );
        })}
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
