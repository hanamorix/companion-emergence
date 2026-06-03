const capitalize = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

type Props = {
  markdown: string;
  persona: string;
};

export function DraftSpacePanel({ markdown, persona }: Props) {
  if (!markdown.trim()) {
    return (
      <aside className="draft-space-panel" role="region">
        <p className="empty">No drafts yet.</p>
      </aside>
    );
  }
  // For v0.0.9, render raw markdown in a pre block. A future iteration can
  // adopt a proper markdown renderer (react-markdown) once the design is
  // settled.
  return (
    <aside className="draft-space-panel" role="region" aria-label="Draft space">
      <h2>Fragments {capitalize(persona)} left while you were away</h2>
      <pre className="draft-space-panel__content">{markdown}</pre>
    </aside>
  );
}
