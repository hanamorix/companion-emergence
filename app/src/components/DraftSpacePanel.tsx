type Props = {
  markdown: string;
};

export function DraftSpacePanel({ markdown }: Props) {
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
      <h2>Fragments Nell left while you were away</h2>
      <pre className="draft-space-panel__content">{markdown}</pre>
    </aside>
  );
}

