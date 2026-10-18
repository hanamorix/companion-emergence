import { useState } from "react";
import { setPersonaModel, type ChatModel } from "../../bridge";
import { MODEL_ID_MAX_LENGTH, isChatModelAlias, isValidChatModel } from "../../lib/modelId";

/** Radio value for the custom-id option; never sent to the bridge (the typed id is). */
const CUSTOM = "__custom__";

interface Props {
  current: ChatModel;
  persona: string;
  onClose: (newModel?: ChatModel) => void;
}

interface ModelOption {
  value: ChatModel;
  label: string;
  hint: string;
}

const OPTIONS: ModelOption[] = [
  {
    value: "sonnet",
    label: "Sonnet",
    hint: "fast, smart, ~$3/M input tokens (recommended)",
  },
  {
    value: "opus",
    label: "Opus",
    hint: "smartest, ~$15/M input tokens (best for deep writing)",
  },
  {
    value: "haiku",
    label: "Haiku",
    hint: "fastest, cheapest, less capable",
  },
];

export function ModelPicker({ current, persona, onClose }: Props) {
  const currentIsAlias = isChatModelAlias(current);
  const [selected, setSelected] = useState<string>(currentIsAlias ? current : CUSTOM);
  const [customText, setCustomText] = useState(currentIsAlias ? "" : current);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isCustom = selected === CUSTOM;
  // Surrounding whitespace is dropped (a pasted id often has some); the rest is validated as written.
  const value = isCustom ? customText.trim() : selected;
  const valid = isValidChatModel(value);
  const canApply = !busy && valid && value !== current;
  const showFormatHint = isCustom && value !== "" && !valid;

  async function onConfirm() {
    setBusy(true);
    setError(null);
    try {
      await setPersonaModel(persona, value);
      onClose(value);
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
            name="model"
            value={o.value}
            checked={selected === o.value}
            onChange={() => setSelected(o.value)}
            style={{ marginTop: 2, flexShrink: 0 }}
          />
          <span>
            <strong>{o.label}</strong>
            {" — "}
            <span style={{ color: "var(--text-mute)" }}>{o.hint}</span>
          </span>
        </label>
      ))}

      <label
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
          name="model"
          value={CUSTOM}
          checked={isCustom}
          onChange={() => setSelected(CUSTOM)}
          style={{ marginTop: 2, flexShrink: 0 }}
        />
        <span>
          <strong>Custom model ID</strong>
          {": "}
          <span style={{ color: "var(--text-mute)" }}>pin a specific version</span>
        </span>
      </label>
      <input
        type="text"
        aria-label="Model ID"
        aria-describedby={showFormatHint ? "model-id-hint" : undefined}
        placeholder="claude-..."
        maxLength={MODEL_ID_MAX_LENGTH}
        value={customText}
        spellCheck={false}
        autoCapitalize="off"
        autoComplete="off"
        onFocus={() => setSelected(CUSTOM)}
        onChange={(e) => {
          setCustomText(e.target.value);
          setSelected(CUSTOM);
        }}
        style={{
          marginLeft: 22,
          padding: "4px 6px",
          fontSize: 11,
          fontFamily: "var(--font-ui)",
          color: "var(--text)",
          background: "transparent",
          border: "1px solid var(--hairline)",
          borderRadius: 4,
        }}
      />
      {showFormatHint && (
        <div
          id="model-id-hint"
          style={{ marginLeft: 22, fontSize: 10.5, color: "var(--text-mute)", lineHeight: 1.45 }}
        >
          Model IDs start with claude- and use lowercase letters, digits, dots and dashes.
        </div>
      )}

      <div
        style={{
          display: "flex",
          gap: 6,
          marginTop: 4,
        }}
      >
        <button
          onClick={onConfirm}
          disabled={!canApply}
          style={{
            flex: 1,
            padding: "6px 10px",
            fontSize: 11,
            fontFamily: "var(--font-ui)",
            background: "color-mix(in srgb, var(--accent) 15%, transparent)",
            color: "var(--text)",
            border: "1px solid color-mix(in srgb, var(--accent) 30%, transparent)",
            borderRadius: 5,
            cursor: canApply ? "pointer" : "not-allowed",
            opacity: canApply ? 1 : 0.55,
          }}
        >
          {busy ? (isCustom ? "Checking..." : "Applying...") : "Apply"}
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
