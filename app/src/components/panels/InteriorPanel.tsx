import { useEffect, useState } from "react";
import type { InteriorEntry, PersonaState } from "../../bridge";
import { PanelShell, SectionLabel } from "../ui";

interface Props {
  state: PersonaState | null;
}

/**
 * Recent Interior — dream / research / heartbeat / reflex narrative
 * paragraphs. Each section shows its summary plus a live "X ago"
 * badge driven by the entry's timestamp so the panel feels alive
 * instead of a static snapshot of the last fire. Sections fired
 * within the last 5 minutes pulse with a small accent dot. Absent
 * sections render nothing rather than "n/a" to match the
 * "silence is meaningful" voice principle.
 */
export function InteriorPanel({ state }: Props) {
  const interior = state?.interior;
  // Tick once a minute so the "ago" labels stay current without a
  // round-trip to the bridge. The state poll itself only refreshes
  // every 5s when something actually changed; the ago label needs
  // to advance even when nothing fires.
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 60_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PanelShell>
      <SectionLabel>Recent Interior</SectionLabel>
      {interior ? (
        <>
          {interior.dream && <Section heading="Dream" entry={interior.dream} />}
          {interior.research && <Section heading="Research" entry={interior.research} />}
          {interior.heartbeat && <Section heading="Heartbeat" entry={interior.heartbeat} />}
          {interior.reflex && <Section heading="Reflex" entry={interior.reflex} />}
          {!interior.dream &&
            !interior.research &&
            !interior.heartbeat &&
            !interior.reflex && (
              <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
                Quiet inside.
              </div>
            )}
        </>
      ) : (
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          No signal yet.
        </div>
      )}
    </PanelShell>
  );
}

function Section({ heading, entry }: { heading: string; entry: InteriorEntry }) {
  const ageInfo = entry.ts ? agoLabel(entry.ts) : null;
  const fresh = ageInfo !== null && ageInfo.seconds < 300; // <5 min
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 4,
        }}
      >
        {fresh && (
          <span
            aria-hidden="true"
            title="just fired"
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "var(--accent)",
              animation: "pulse 1.6s ease-in-out infinite",
              flexShrink: 0,
            }}
          />
        )}
        <div
          style={{
            fontSize: "9.5px",
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            fontFamily: "var(--font-disp)",
            flex: 1,
          }}
        >
          {heading}
        </div>
        {ageInfo && (
          <div
            style={{
              fontSize: "9.5px",
              color: "var(--text-mute)",
              fontFamily: "var(--font-disp)",
              letterSpacing: "0.04em",
              fontStyle: "italic",
            }}
            title={entry.ts ?? undefined}
          >
            {ageInfo.label}
          </div>
        )}
      </div>
      <div
        style={{
          fontSize: 11,
          color: "var(--text-mid)",
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
        }}
      >
        {renderInlineMarkdown(entry.summary)}
      </div>
    </div>
  );
}

/**
 * Format the gap between ``ts`` and now as a short human label
 * ("just now", "5m ago", "2h ago", "3d ago"). Returns null when
 * ts is unparseable so the caller can skip rendering.
 */
function agoLabel(iso: string): { label: string; seconds: number } | null {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  const seconds = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (seconds < 30) return { label: "just now", seconds };
  if (seconds < 3600) return { label: `${Math.round(seconds / 60)}m ago`, seconds };
  const hours = Math.round(seconds / 3600);
  if (hours < 24) return { label: `${hours}h ago`, seconds };
  return { label: `${Math.round(hours / 24)}d ago`, seconds };
}

// Reflex/dream summaries arrive with single-asterisk italic markers
// (`*setting line*`). Render them as <em> so the cards read cleanly
// instead of leaking raw asterisks.
function renderInlineMarkdown(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const re = /\*([^*\n]+)\*/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(<em key={key++}>{match[1]}</em>);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length ? parts : [text];
}
