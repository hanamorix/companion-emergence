import { useState } from "react";

export type VoiceEditProposal = {
  auditId: string;
  oldText: string;
  newText: string;
  rationale: string;
  evidence: string[];
  voiceTemplate: string;
};

type Props = {
  proposal: VoiceEditProposal;
  onAccept: (auditId: string, withEdits: string | null) => void;
  onReject: (auditId: string) => void;
};

export function VoiceEditPanel({ proposal, onAccept, onReject }: Props) {
  const [editMode, setEditMode] = useState(false);
  const [editedText, setEditedText] = useState(proposal.newText);

  const lines = proposal.voiceTemplate.split("\n");
  const targetIdx = lines.findIndex((l) => l.trim() === proposal.oldText.trim());

  return (
    <aside className="voice-edit-panel" role="dialog" aria-label="Voice edit proposal">
      <h2>Nell proposed an edit to her voice</h2>
      <p className="voice-edit-panel__rationale">{proposal.rationale}</p>
      <p className="voice-edit-panel__evidence">
        Evidence: {proposal.evidence.join(", ")}
      </p>

      <pre className="voice-edit-panel__diff">
        {lines.map((line, i) => {
          if (i === targetIdx) {
            return (
              <div key={i}>
                <div className="diff-line diff-line--remove">- {line}</div>
                <div className="diff-line diff-line--add">+ {proposal.newText}</div>
              </div>
            );
          }
          return (
            <div key={i} className="diff-line diff-line--context">{"  "}{line}</div>
          );
        })}
      </pre>

      <div className="voice-edit-panel__actions">
        {!editMode ? (
          <>
            <button onClick={() => onAccept(proposal.auditId, null)}>Accept</button>
            <button onClick={() => setEditMode(true)}>Accept with edits</button>
            <button onClick={() => onReject(proposal.auditId)}>Reject</button>
          </>
        ) : (
          <>
            <textarea
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
              aria-label="Edit the proposed new text"
            />
            <button onClick={() => onAccept(proposal.auditId, editedText)}>Confirm</button>
            <button onClick={() => setEditMode(false)}>Cancel</button>
          </>
        )}
      </div>
    </aside>
  );
}
