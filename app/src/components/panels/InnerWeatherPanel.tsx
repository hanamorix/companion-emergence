import type { PersonaState } from "../../bridge";
import { Bar, PanelShell, SectionLabel } from "../ui";

const TOP_N_EMOTIONS = 8;
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
 */
export function InnerWeatherPanel({ state }: Props) {
  return (
    <PanelShell>
      <SectionLabel>Inner Weather</SectionLabel>
      <SectionLabel>Emotional</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {state?.emotions ? (
          Object.entries(state.emotions)
            .slice(0, TOP_N_EMOTIONS)
            .map(([name, value]) => <Bar key={name} label={name} value={value} max={10} />)
        ) : (
          <Empty />
        )}
      </div>
      <SectionLabel>Body</SectionLabel>
      {state?.body ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
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

function Empty() {
  return (
    <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
      no signal yet
    </div>
  );
}
