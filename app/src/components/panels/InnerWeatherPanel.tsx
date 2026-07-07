import { useEffect, useRef } from "react";
import type { PersonaState } from "../../bridge";
import { Bar, PanelShell, SectionLabel } from "../ui";

const TOP_N_EMOTIONS = 7;
const DELTA_ARROW_THRESHOLD = 0.2;
const BODY_HIGHLIGHT_FIELDS: Array<{ key: "energy" | "temperature"; label: string; max: number }> = [
  { key: "energy", label: "Energy", max: 10 },
  { key: "temperature", label: "Temp", max: 9 },
];

interface Props {
  state: PersonaState | null;
}

/**
 * Inner Weather — the most-frequent panel: top emotions sorted desc
 * with a body-energy/temp summary at the bottom. Matches mockup
 * nell_face_example_1.png.
 *
 * Delta arrows (▲/▼) are purely visual: a ref remembers the previous
 * poll's emotion values so a bar can show whether it moved since the
 * last render. No backend involvement.
 */
export function InnerWeatherPanel({ state }: Props) {
  const prevEmotionsRef = useRef<Record<string, number> | null>(null);
  const emotions = state?.emotions ?? null;
  const previous = prevEmotionsRef.current;
  useEffect(() => {
    prevEmotionsRef.current = emotions;
  }, [emotions]);

  return (
    <PanelShell>
      <SectionLabel>Inner Weather</SectionLabel>
      <SectionLabel>Emotional</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
        {emotions ? (
          Object.entries(emotions)
            .slice(0, TOP_N_EMOTIONS)
            .map(([name, value]) => (
              <Bar
                key={name}
                label={name}
                value={value}
                max={10}
                arrow={deltaArrow(value, previous?.[name])}
              />
            ))
        ) : (
          <Empty />
        )}
      </div>
      <SectionLabel>Body</SectionLabel>
      {state?.body ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          {BODY_HIGHLIGHT_FIELDS.map(({ key, label, max }) => (
            <Bar
              key={key}
              label={label}
              value={state.body![key]}
              max={max}
              formatValue={(v) => `${v.toFixed(0)}/${max}`}
            />
          ))}
        </div>
      ) : (
        <Empty />
      )}
    </PanelShell>
  );
}

/** ▲ when the value climbed, ▼ when it fell, null when the delta is too
 *  small to matter or there's no prior poll to compare against yet. */
function deltaArrow(
  current: number,
  prev: number | undefined,
): { glyph: "▲" | "▼"; color: string } | null {
  if (prev === undefined) return null;
  const delta = current - prev;
  if (Math.abs(delta) <= DELTA_ARROW_THRESHOLD) return null;
  return delta > 0 ? { glyph: "▲", color: "#5fbe8b" } : { glyph: "▼", color: "#e07a6a" };
}

function Empty() {
  return (
    <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
      No signal yet.
    </div>
  );
}
