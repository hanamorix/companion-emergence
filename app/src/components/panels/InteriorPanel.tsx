import type { PersonaState } from "../../bridge";
import { PanelShell, SectionLabel } from "../ui";

interface Props {
  state: PersonaState | null;
}

/**
 * Recent Interior — dream / research / heartbeat / reflex narrative
 * paragraphs. Matches mockup nell_face_example_3.png. Each section
 * shows its theme; absent sections render nothing rather than "n/a"
 * to match the "silence is meaningful" voice principle.
 */
export function InteriorPanel({ state }: Props) {
  const interior = state?.interior;
  return (
    <PanelShell>
      <SectionLabel>Recent Interior</SectionLabel>
      {interior ? (
        <>
          {interior.dream && (
            <Section heading="Dream" body={interior.dream} />
          )}
          {interior.research && (
            <Section heading="Research" body={interior.research} />
          )}
          {interior.heartbeat && (
            <Section heading="Heartbeat" body={interior.heartbeat} />
          )}
          {interior.reflex && (
            <Section heading="Reflex" body={interior.reflex} />
          )}
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

function Section({ heading, body }: { heading: string; body: string }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          fontSize: "9.5px",
          color: "var(--text-mute)",
          textTransform: "uppercase",
          letterSpacing: "0.12em",
          fontFamily: "var(--font-disp)",
          marginBottom: 4,
        }}
      >
        {heading}
      </div>
      <div
        style={{
          fontSize: 11,
          color: "var(--text-mid)",
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
        }}
      >
        {renderInlineMarkdown(body)}
      </div>
    </div>
  );
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
