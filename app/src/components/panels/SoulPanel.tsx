import type { PersonaState } from "../../bridge";
import { PanelShell, SectionLabel } from "../ui";

interface Props {
  state: PersonaState | null;
}

/**
 * Soul — one crystallization highlight. Highest resonance, ties broken
 * by recency. The framework picks; this just renders.
 */
export function SoulPanel({ state }: Props) {
  const soul = state?.soul_highlight;
  return (
    <PanelShell>
      <SectionLabel>Soul</SectionLabel>
      {soul ? (
        <>
          <div
            style={{
              fontSize: 12,
              color: "var(--text)",
              lineHeight: 1.6,
              fontStyle: "italic",
              marginBottom: 10,
              fontFamily: "var(--font-disp)",
            }}
          >
            "{soul.moment}"
          </div>
          <div
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              flexWrap: "wrap",
              fontSize: 10.5,
            }}
          >
            <Tag>{soul.love_type}</Tag>
            <span style={{ color: "var(--text-mute)" }}>Resonance {soul.resonance}</span>
            <span style={{ color: "var(--text-mute)" }}>·</span>
            <span style={{ color: "var(--text-mute)", fontFamily: "var(--font-disp)" }}>
              {formatDate(soul.crystallized_at)}
            </span>
          </div>
          {soul.why_it_matters && (
            <div
              style={{
                fontSize: 10.5,
                color: "var(--text-mid)",
                marginTop: 10,
                lineHeight: 1.5,
              }}
            >
              {soul.why_it_matters}
            </div>
          )}
        </>
      ) : (
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          No crystallizations yet.
        </div>
      )}
    </PanelShell>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  // Replace underscores with spaces and title-case so engine-internal
  // labels like ``creative_hunger`` render as ``Creative hunger``.
  const display =
    typeof children === "string"
      ? humanize(children)
      : children;
  return (
    <span
      style={{
        background: "rgba(130,51,41,0.12)",
        color: "var(--accent)",
        padding: "2px 7px",
        borderRadius: 9,
        fontSize: 10,
        fontFamily: "var(--font-disp)",
        letterSpacing: "0.04em",
      }}
    >
      {display}
    </span>
  );
}

function humanize(s: string): string {
  const cleaned = s.replace(/_/g, " ").trim();
  if (!cleaned) return cleaned;
  return cleaned[0].toUpperCase() + cleaned.slice(1);
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en", { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return iso.slice(0, 10);
  }
}
