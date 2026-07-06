import { useState } from "react";
import type { PersonaState } from "../bridge";
import { InnerWeatherPanel } from "./panels/InnerWeatherPanel";
import { BodyPanel } from "./panels/BodyPanel";
import { FeedPanel } from "./panels/FeedPanel";
import { SoulPanel } from "./panels/SoulPanel";
import { ConnectionPanel } from "./panels/ConnectionPanel";
import { GalleryPanel } from "./panels/GalleryPanel";
import { AttunementPanel } from "./panels/AttunementPanel";
import { KindledLinksPanel } from "./panels/KindledLinksPanel";

type Tab = "weather" | "body" | "interior" | "soul" | "connection" | "gallery" | "attunement" | "kindled_link";

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
  { id: "interior", label: "Inner Life", icon: "✦" },
  { id: "soul", label: "Soul", icon: "❀" },
  { id: "attunement", label: "Attunement", icon: "∿" },
  { id: "kindled_link", label: "Kindled Links", icon: "⌁" },
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
        // panel (272px) + gap (10px) + rail (~48px) = 330px reserved width.
        // Rail anchors to the left edge; panel expands rightward so toggle
        // icons never sit on top of stats content in small windows.
        width: 330,
        display: "flex",
        justifyContent: "flex-start",
        alignItems: "flex-start",
      }}
    >
      {/* Rail — anchored to left edge, always painted on top */}
      <div
        className="glass"
        style={{
          position: "relative",
          zIndex: 2,
          display: "flex",
          flexDirection: "column",
          gap: 5,
          padding: 7,
          borderRadius: 18,
          backdropFilter: "blur(28px) saturate(1.5)",
          WebkitBackdropFilter: "blur(28px) saturate(1.5)",
          boxShadow: "var(--shadow-mid)",
          flexShrink: 0,
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
                width: 36,
                height: 36,
                borderRadius: 11,
                background: active ? "var(--accent)" : "transparent",
                color: active ? "#ffffff" : "var(--text-mid)",
                border: "none",
                boxShadow: active
                  ? "0 6px 16px color-mix(in srgb, var(--accent) 40%, transparent)"
                  : "none",
                fontSize: 15,
                transition: "all 0.2s ease",
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
      {/* Panel — opens to the left of the rail (rail 48 + gap 6 = 54) */}
      {tab !== null && (
        <div style={{ position: "absolute", left: 54, top: 0, zIndex: 1 }}>
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
      return <FeedPanel state={state} />;
    case "soul":
      return <SoulPanel state={state} />;
    case "attunement":
      return <AttunementPanel persona={opts.persona} />;
    case "kindled_link":
      return <KindledLinksPanel persona={opts.persona} />;
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
