import { useEffect, useState } from "react";
import type { AttunementPayload, LearnedPattern } from "../../bridge";
import { fetchAttunement } from "../../bridge";
import { errString } from "../../lib/errString";
import { PanelShell, SectionLabel } from "../ui";

interface Props {
  persona: string;
}

const MATURITY_ORDER: Record<string, number> = {
  known: 0,
  forming: 1,
  falsified: 2,
};

const capitalize = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export function AttunementPanel({ persona }: Props) {
  const [payload, setPayload] = useState<AttunementPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const p = await fetchAttunement(persona);
        if (!cancelled) {
          setPayload(p);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(errString(e));
      }
    };
    load();
    const id = setInterval(load, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [persona]);

  if (error) {
    return (
      <PanelShell>
        <SectionLabel>Attunement</SectionLabel>
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          {error}
        </div>
      </PanelShell>
    );
  }

  if (payload === null) {
    return (
      <PanelShell>
        <SectionLabel>Attunement</SectionLabel>
      </PanelShell>
    );
  }

  const { current_read, learned_patterns, backfill } = payload;

  const surfaceable = learned_patterns
    .filter((p: LearnedPattern) => p.maturity !== "immature")
    .sort(
      (a: LearnedPattern, b: LearnedPattern) =>
        (MATURITY_ORDER[a.maturity] ?? 99) - (MATURITY_ORDER[b.maturity] ?? 99),
    );

  const backfillRunning = backfill !== null && backfill.status !== "complete";
  const hasContent = current_read !== null || surfaceable.length > 0 || backfillRunning;

  return (
    <PanelShell>
      <SectionLabel>Attunement</SectionLabel>

      {!hasContent && (
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          Nothing yet — {capitalize(persona)} is just getting started with you.
        </div>
      )}

      {current_read !== null && (
        <section style={{ marginBottom: 14 }}>
          <div
            style={{
              fontSize: "9.5px",
              color: "var(--text-mute)",
              textTransform: "uppercase",
              letterSpacing: "0.12em",
              fontFamily: "var(--font-disp)",
              marginBottom: 6,
            }}
          >
            Right now
          </div>
          <div
            style={{
              paddingLeft: 13,
              borderLeft: "1px solid rgba(191, 184, 173, 0.10)",
              fontSize: 11,
              color: "var(--text-mid)",
              lineHeight: 1.55,
            }}
          >
            <p style={{ margin: "0 0 4px" }}>
              She sounds{" "}
              <strong style={{ color: "var(--text)" }}>{current_read.tone_label}</strong> —{" "}
              {current_read.tone_justification}.
            </p>
            <p style={{ margin: "0 0 4px" }}>
              Her cadence is{" "}
              <strong style={{ color: "var(--text)" }}>{current_read.cadence_label}</strong> —{" "}
              {current_read.cadence_justification}.
            </p>
            {current_read.predicted_arc_shape && (
              <p style={{ margin: 0 }}>
                Where this seems to be heading: {current_read.predicted_arc_shape}.
              </p>
            )}
          </div>
        </section>
      )}

      {surfaceable.length > 0 && (
        <section>
          <div
            style={{
              fontSize: "9.5px",
              color: "var(--text-mute)",
              textTransform: "uppercase",
              letterSpacing: "0.12em",
              fontFamily: "var(--font-disp)",
              marginBottom: 6,
            }}
          >
            What she&#39;s come to know
          </div>
          <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
            {surfaceable.map((p: LearnedPattern) => (
              <PatternItem key={p.id} pattern={p} />
            ))}
          </ul>
        </section>
      )}

      {backfillRunning && backfill !== null && (
        <div
          style={{
            padding: "6px 8px",
            background: "rgba(200, 152, 144, 0.12)",
            border: "1px solid rgba(200, 152, 144, 0.25)",
            borderRadius: 4,
            marginTop: 12,
            fontSize: 10.5,
            color: "var(--text-mid)",
            fontFamily: "var(--font-disp)",
            lineHeight: 1.5,
          }}
        >
          {capitalize(persona)} is getting to know you.{" "}
          <span>
            {backfill.sampled_windows ?? 0} / {backfill.total_windows ?? 0}
          </span>{" "}
          windows processed.
        </div>
      )}
    </PanelShell>
  );
}

function PatternItem({ pattern }: { pattern: LearnedPattern }) {
  const isFalsified = pattern.maturity === "falsified";
  return (
    <li
      data-maturity={pattern.maturity}
      style={{
        marginBottom: 10,
        paddingLeft: 13,
        borderLeft: "1px solid rgba(191, 184, 173, 0.10)",
      }}
    >
      <div
        style={{
          textDecoration: isFalsified ? "line-through" : undefined,
          opacity: isFalsified ? 0.55 : 1,
          fontSize: 11,
          color: "var(--text-mid)",
          lineHeight: 1.5,
        }}
      >
        {pattern.description}
      </div>
      <div
        style={{
          display: "flex",
          gap: 6,
          marginTop: 2,
          fontSize: "9.5px",
          color: "var(--text-mute)",
          fontFamily: "var(--font-disp)",
        }}
      >
        <span
          style={{
            background: "rgba(130, 51, 41, 0.10)",
            padding: "1px 5px",
            borderRadius: 3,
          }}
        >
          {pattern.category}
        </span>
        <span
          style={{
            background: isFalsified
              ? "rgba(130, 51, 41, 0.06)"
              : pattern.maturity === "known"
                ? "rgba(130, 51, 41, 0.18)"
                : "rgba(130, 51, 41, 0.10)",
            padding: "1px 5px",
            borderRadius: 3,
          }}
        >
          {pattern.maturity}
        </span>
      </div>
      <div
        style={{
          fontSize: "9.5px",
          color: "var(--text-mute)",
          fontFamily: "var(--font-disp)",
          marginTop: 3,
        }}
      >
        First noticed: {agoLabel(pattern.first_seen_at)}
        {pattern.last_addressed_at !== null && (
          <> &middot; Named: {agoLabel(pattern.last_addressed_at)}</>
        )}
      </div>
      {pattern.examples.length > 0 && (
        <details style={{ marginTop: 4 }}>
          <summary
            style={{
              fontSize: "9.5px",
              color: "var(--text-mute)",
              fontFamily: "var(--font-disp)",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            Examples ({pattern.examples.length})
          </summary>
          <ul
            style={{
              listStyle: "none",
              margin: "4px 0 0",
              padding: 0,
              display: "flex",
              flexDirection: "column",
              gap: 3,
            }}
          >
            {pattern.examples.map((ex, i) => (
              <li
                key={i}
                style={{
                  fontSize: 10.5,
                  color: "var(--text-mute)",
                  fontStyle: "italic",
                  fontFamily: "var(--font-disp)",
                  lineHeight: 1.45,
                }}
              >
                {ex}
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}

function agoLabel(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const seconds = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (seconds < 30) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  const hours = Math.round(seconds / 3600);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
