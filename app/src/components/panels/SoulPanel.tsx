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
              fontSize: 13.5,
              color: "var(--text)",
              lineHeight: 1.65,
              fontStyle: "italic",
              fontFamily: "var(--font-disp)",
              background: "color-mix(in srgb, var(--accent) 9%, transparent)",
              border: "1px solid color-mix(in srgb, var(--accent) 24%, transparent)",
              borderRadius: 16,
              padding: "12px 14px",
              marginBottom: 10,
            }}
          >
            "{soul.moment}"
          </div>
          <div
            style={{
              fontSize: 9.5,
              fontStyle: "italic",
              color: "var(--text-mute)",
              marginBottom: 10,
            }}
          >
            crystallized · protected
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
            <span style={{ color: "var(--text-mute)" }}>
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
        background: "color-mix(in srgb, var(--accent) 22%, transparent)",
        color: "var(--accent-text)",
        padding: "2px 7px",
        borderRadius: 9,
        fontSize: 9.5,
        fontWeight: 700,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
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
