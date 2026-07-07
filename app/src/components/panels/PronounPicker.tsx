import { useState } from "react";
import { setPersonaPronouns, type PronounPreset } from "../../bridge";

interface Props {
  current: PronounPreset;
  persona: string;
  onClose: (next?: PronounPreset) => void;
}

interface PronounOption {
  value: PronounPreset;
  label: string;
}

const OPTIONS: PronounOption[] = [
  { value: "she/her", label: "she/her" },
  { value: "he/him", label: "he/him" },
  { value: "they/them", label: "they/them" },
];

export function PronounPicker({ current, persona, onClose }: Props) {
  const [selected, setSelected] = useState<PronounPreset>(current);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onConfirm() {
    setBusy(true);
    setError(null);
    try {
      await setPersonaPronouns(persona, selected);
      onClose(selected);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        padding: "8px 0",
      }}
    >
      {OPTIONS.map((o) => (
        <label
          key={o.value}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 8,
            cursor: "pointer",
            fontSize: 11,
            lineHeight: 1.5,
            color: "var(--text)",
            fontFamily: "var(--font-ui)",
          }}
        >
          <input
            type="radio"
            name="pronouns"
            value={o.value}
            checked={selected === o.value}
            onChange={() => setSelected(o.value)}
            style={{ marginTop: 2, flexShrink: 0 }}
          />
          <span>
            <strong>{o.label}</strong>
          </span>
        </label>
      ))}

      <div
        style={{
          display: "flex",
          gap: 6,
          marginTop: 4,
        }}
      >
        <button
          onClick={onConfirm}
          disabled={busy || selected === current}
          style={{
            flex: 1,
            padding: "6px 10px",
            fontSize: 11,
            fontFamily: "var(--font-ui)",
            background: "color-mix(in srgb, var(--accent) 15%, transparent)",
            color: "var(--text)",
            border: "1px solid color-mix(in srgb, var(--accent) 30%, transparent)",
            borderRadius: 5,
            cursor: busy || selected === current ? "not-allowed" : "pointer",
            opacity: busy || selected === current ? 0.55 : 1,
          }}
        >
          {busy ? "Applying..." : "Apply"}
        </button>
        <button
          onClick={() => onClose()}
          disabled={busy}
          style={{
            padding: "6px 10px",
            fontSize: 11,
            fontFamily: "var(--font-ui)",
            background: "transparent",
            color: "var(--text-mute)",
            border: "1px solid var(--hairline)",
            borderRadius: 5,
            cursor: busy ? "not-allowed" : "pointer",
          }}
        >
          Cancel
        </button>
      </div>

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
