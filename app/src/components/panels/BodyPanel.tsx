import type { PersonaState } from "../../bridge";
import { Bar, Divider, PanelShell, SectionLabel } from "../ui";

interface Props {
  state: PersonaState | null;
}

/**
 * Body — the full body block: energy/temp/exhaustion + body emotions
 * (arousal/desire/climax/touch_hunger/comfort_seeking/rest_need) +
 * session/contact metadata. Matches mockup nell_face_example_2.png.
 */
export function BodyPanel({ state }: Props) {
  const body = state?.body;
  return (
    <PanelShell>
      <SectionLabel>Body</SectionLabel>
      {body ? (
        <>
          <Bar label="Energy" value={body.energy} max={10} formatValue={(v) => `${v}/10`} />
          <Bar label="Temp" value={body.temperature} max={9} formatValue={(v) => `${v}/9`} />
          <Bar
            label="Exhaust"
            value={body.exhaustion}
            max={10}
            formatValue={(v) => `${v}/10`}
          />
          <Divider />
          <SectionLabel>Body Emotions</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {Object.entries(body.body_emotions)
              .filter(([, v]) => v > 0.4)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 5)
              .map(([name, value]) => (
                <Bar key={name} label={name} value={value} max={10} />
              ))}
            {Object.values(body.body_emotions).every((v) => v <= 0.4) && (
              <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
                Quiet.
              </div>
            )}
          </div>
          <Divider />
          <KeyValue label="Session" value={`${body.session_hours.toFixed(1)}h`} />
          <KeyValue
            label="Contact"
            value={
              body.days_since_contact >= 1
                ? `${body.days_since_contact.toFixed(1)}d`
                : `${(body.days_since_contact * 24).toFixed(1)}h`
            }
          />
        </>
      ) : (
        <div style={{ fontSize: 11, color: "var(--text-mute)", fontStyle: "italic" }}>
          Body offline.
        </div>
      )}
    </PanelShell>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        fontSize: 11.5,
        marginBottom: 3,
      }}
    >
      <span style={{ color: "var(--text-mid)" }}>{label}</span>
      <span style={{ color: "var(--text)", fontFamily: "var(--font-disp)" }}>{value}</span>
    </div>
  );
}
