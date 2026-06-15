import { useState } from "react";
import { setPersonaNotes } from "../../bridge";

interface Props {
  /** Active persona — display name + endpoint target. */
  persona: string;
  /** Whether notes are currently enabled (from persona state). */
  enabled: boolean;
  /** The resolved per-OS notes folder when enabled, else null. */
  folder: string | null;
}

/**
 * One-time consent toggle: "Let {persona} leave me notes". When the user
 * enables it, the bridge resolves + creates the per-OS notes folder and
 * returns its path; we surface that path read-only so the user knows where
 * the notes will land. The user never picks a path — the system does.
 */
export function NotesToggle({ persona, enabled, folder }: Props) {
  const [on, setOn] = useState(enabled);
  const [localFolder, setLocalFolder] = useState<string | null>(folder);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onToggle() {
    const next = !on;
    setBusy(true);
    setError(null);
    try {
      const res = await setPersonaNotes(persona, next);
      setOn(res.enabled);
      setLocalFolder(res.folder);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        padding: "4px 0",
      }}
    >
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          cursor: busy ? "not-allowed" : "pointer",
          fontSize: 11,
          lineHeight: 1.5,
          color: "var(--text)",
          fontFamily: "var(--font-ui)",
        }}
      >
        <input
          type="checkbox"
          checked={on}
          disabled={busy}
          onChange={onToggle}
          style={{ flexShrink: 0 }}
        />
        <span>
          Let <strong>{persona}</strong> leave me notes
        </span>
      </label>

      {on && localFolder && (
        <div
          style={{
            fontSize: 10.5,
            color: "var(--text-mute)",
            lineHeight: 1.5,
            wordBreak: "break-word",
          }}
        >
          Notes will be left in:{" "}
          <span style={{ color: "var(--text)", fontWeight: 500 }}>{localFolder}</span>
        </div>
      )}

      {error && (
        <div
          role="alert"
          style={{
            fontSize: 10.5,
            color: "var(--crimson)",
            lineHeight: 1.45,
            wordBreak: "break-word",
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
