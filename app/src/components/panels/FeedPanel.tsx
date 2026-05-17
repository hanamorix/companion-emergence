import { useEffect, useState } from "react";
import type { FeedEntry, FeedEntryType, PersonaState } from "../../bridge";
import { fetchPersonaFeed } from "../../bridge";
import { PanelShell, SectionLabel } from "../ui";

interface Props {
  state: PersonaState | null;
}

const TYPE_LABEL: Record<FeedEntryType, string> = {
  dream: "Dream",
  research: "Research",
  soul: "Soul",
  outreach: "Outreach",
  voice_edit: "Voice edit",
};

const TYPE_DOT: Record<FeedEntryType, string> = {
  dream: "#6b95b8",
  research: "#b89c6b",
  soul: "#b87fa3",
  outreach: "#823329",   // project accent
  voice_edit: "#7fa37f",
};

/**
 * Visible inner life feed — chronological journal across 5 source streams
 * (dream / research / soul / outreach / voice_edit). Replaces the legacy
 * InteriorPanel snapshot. Polls /persona/feed on a 5s cadence whenever
 * persona state is present; renders Layout B from the v0.0.13-alpha.2
 * spec (colored type-dot per engine, hairline-rule indented body, fresh
 * pulse for <5min items, 1-minute ago-label refresh).
 */
export function FeedPanel({ state }: Props) {
  const [entries, setEntries] = useState<FeedEntry[] | null>(null);
  const [, force] = useState(0);

  // Initial fetch + 5s poll whenever state is present.
  useEffect(() => {
    if (state == null) return;
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const next = await fetchPersonaFeed(state.persona);
        if (!cancelled) setEntries(next);
      } catch {
        // Mirror InteriorPanel: swallow + leave previous state visible.
      }
    };
    fetchOnce();
    const id = setInterval(fetchOnce, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [state]);

  // One-minute tick keeps "ago" labels current.
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 60_000);
    return () => clearInterval(id);
  }, []);

  return (
    <PanelShell>
      <SectionLabel>Inner life</SectionLabel>
      {state == null ? (
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          No signal yet.
        </div>
      ) : entries === null ? null : entries.length === 0 ? (
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          Quiet inside.
        </div>
      ) : (
        entries.map((entry, i) => <FeedItem key={`${entry.ts}-${i}`} entry={entry} />)
      )}
    </PanelShell>
  );
}

function FeedItem({ entry }: { entry: FeedEntry }) {
  const ageInfo = agoLabel(entry.ts);
  const fresh = ageInfo !== null && ageInfo.seconds < 300; // <5 min
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          marginBottom: 4,
        }}
      >
        {fresh && (
          <span
            data-fresh="true"
            aria-hidden="true"
            title="just fired"
            style={{
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "var(--accent)",
              animation: "pulse 1.6s ease-in-out infinite",
              flexShrink: 0,
            }}
          />
        )}
        <span
          aria-hidden="true"
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: TYPE_DOT[entry.type],
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: "9.5px",
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.14em",
            fontFamily: "var(--font-disp)",
            flex: 1,
          }}
        >
          {TYPE_LABEL[entry.type]}
        </span>
        {ageInfo && (
          <span
            style={{
              fontSize: "9.5px",
              color: "var(--text-mute)",
              fontFamily: "var(--font-disp)",
              fontStyle: "italic",
              letterSpacing: "0.04em",
            }}
            title={entry.ts}
          >
            {ageInfo.label}
          </span>
        )}
      </div>
      <div
        style={{
          paddingLeft: 13,
          borderLeft: "1px solid rgba(191, 184, 173, 0.10)",
          fontSize: 11,
          color: "var(--text-mid)",
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
        }}
      >
        <em
          style={{
            color: "var(--text-mid)",
            fontFamily: "var(--font-disp)",
          }}
        >
          {entry.opener}
        </em>{" "}
        {renderInlineMarkdown(entry.body)}
      </div>
    </div>
  );
}

/** Format the gap between `iso` and now as a short human label. */
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

/** Render single-asterisk italic markers (`*foo*`) as <em>. */
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
