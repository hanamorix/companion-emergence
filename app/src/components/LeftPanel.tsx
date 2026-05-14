import { useState } from "react";
import type { PersonaState } from "../bridge";
import { InnerWeatherPanel } from "./panels/InnerWeatherPanel";
import { BodyPanel } from "./panels/BodyPanel";
import { InteriorPanel } from "./panels/InteriorPanel";
import { SoulPanel } from "./panels/SoulPanel";
import { ConnectionPanel } from "./panels/ConnectionPanel";
import { GalleryPanel } from "./panels/GalleryPanel";

type Tab = "weather" | "body" | "interior" | "soul" | "connection" | "gallery";

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
  { id: "gallery", label: "Gallery", icon: "◫" },
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
  // Reserve LeftPanel's full width (panel + gap + rail) regardless of
  // whether a panel is open. The rail sits at the right edge of that
  // reserved area; the panel slots into the left side when open.
  // Without this reservation the parent flex centered everything
  // around the narrow rail-only width, then a popped-out panel
  // overflowed the window edge to the left. With it, opening a
  // panel only paints — avatar + chat positions stay locked.
  return (
    <div
      style={{
        position: "relative",
        // panel (220) + gap (24) + rail (≈38) — gap 24 (was 12) so the
        // panel's right edge is visibly clear of the rail icons.
        width: 282,
        display: "flex",
        justifyContent: "flex-end",
        alignItems: "flex-start",
      }}
    >
      {tab !== null && (
        // The panel pops out to the left of the rail. zIndex < the rail's
        // explicit zIndex so the rail icons always paint on top — without
        // this the absolute-positioned panel was covering the rail in
        // document order even though they didn't overlap horizontally.
        <div style={{ position: "absolute", left: 0, top: 0, zIndex: 1 }}>
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
          position: "relative",
          zIndex: 2,
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
    case "gallery":
      return <GalleryPanel persona={opts.persona} />;
  }
}
