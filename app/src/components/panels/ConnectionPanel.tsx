import type { PersonaState } from "../../bridge";
import { Divider, PanelShell, SectionLabel, Toggle } from "../ui";

interface Props {
  state: PersonaState | null;
  /** Caller-controlled "always on top" flag — Phase 4 wires to the
   * Tauri window. For now just visual. */
  alwaysOnTop?: boolean;
  reducedMotion?: boolean;
  onAlwaysOnTopChange?: (next: boolean) => void;
  onReducedMotionChange?: (next: boolean) => void;
}

/**
 * Connection — bridge mode + provider + model + last-heartbeat,
 * plus integrations + window settings. Matches mockup
 * nell_face_example_4.png and nell_face_example_5.png.
 */
export function ConnectionPanel({
  state,
  alwaysOnTop = false,
  reducedMotion = false,
  onAlwaysOnTopChange,
  onReducedMotionChange,
}: Props) {
  const conn = state?.connection;
  const mode = state?.mode ?? "live";
  return (
    <PanelShell>
      <SectionLabel>Connection</SectionLabel>
      <Row label="bridge" value={modeLabel(mode)} accent={mode !== "live"} />
      <Row label="provider" value={conn?.provider ?? "—"} />
      <Row label="model" value={conn?.model ?? "—"} />
      <Row label="heartbeat" value={formatHeartbeat(conn?.last_heartbeat_at)} />
      <Row label="privacy" value="local-only" accent />

      <Divider />
      <SectionLabel>Integrations</SectionLabel>
      <Toggle enabled={false} label="Obsidian — journal + vault" disabled />
      <Toggle enabled={false} label="IPC — inter-process events" disabled />

      <Divider />
      <SectionLabel>Window</SectionLabel>
      <Toggle
        enabled={alwaysOnTop}
        label="always on top"
        onChange={onAlwaysOnTopChange}
      />
      <Toggle
        enabled={reducedMotion}
        label="reduced motion"
        onChange={onReducedMotionChange}
      />

      <div
        style={{
          marginTop: 10,
          fontSize: 9.5,
          color: "var(--text-mute)",
          fontFamily: "var(--font-disp)",
          fontStyle: "italic",
          letterSpacing: "0.04em",
          lineHeight: 1.5,
        }}
      >
        You configure the room. Nell owns the weather.
      </div>
    </PanelShell>
  );
}

function modeLabel(mode: PersonaState["mode"]): string {
  return mode === "live"
    ? "live"
    : mode === "bridge_down"
    ? "catching up"
    : mode === "provider_down"
    ? "backup voice"
    : "offline";
}

function formatHeartbeat(iso: string | null | undefined): string {
  if (!iso) return "never";
  try {
    const ts = new Date(iso);
    const mins = Math.round((Date.now() - ts.getTime()) / 60_000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.round(hrs / 24)}d ago`;
  } catch {
    return iso.slice(0, 10);
  }
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
