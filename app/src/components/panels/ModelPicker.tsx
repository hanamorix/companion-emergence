import { useState } from "react";
import { setPersonaModelFull, type ChatModel } from "../../bridge";
import { MODEL_ID_MAX_LENGTH, isChatModelAlias, isValidChatModel } from "../../lib/modelId";

/** Radio value for the custom-id option; never sent to the bridge (the typed id is). */
const CUSTOM = "__custom__";

interface Props {
  /** The model the chat job runs now (the bridge's effective id). */
  current: ChatModel;
  /** The word that model means (the bridge's `model_choice`): highlights its radio; null for a custom id, which
   *  opens the custom field instead. */
  choice: string | null;
  persona: string;
  /** `info` is only passed when the bridge could not confirm the model (#440). */
  onClose: (newModel?: ChatModel, info?: ModelApplyInfo) => void;
  /** Let Apply be pressed for the model that is already current (an unsupported setting fell back to it, #446). */
  allowReapply?: boolean;
}

export interface ModelApplyInfo {
  /** false: the bridge could not confirm the model (it was saved anyway). */
  checked?: false;
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

export function ModelPicker({ current, choice, persona, onClose, allowReapply = false }: Props) {
  const choiceIsWord = choice !== null && isChatModelAlias(choice);
  const [selected, setSelected] = useState<string>(choiceIsWord ? choice : CUSTOM);
  const [customText, setCustomText] = useState(choiceIsWord ? "" : current);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isCustom = selected === CUSTOM;
  // Surrounding whitespace is dropped (a pasted id often has some); the rest is validated as written.
  const value = isCustom ? customText.trim() : selected;
  const valid = isValidChatModel(value);
  // allowReapply: the shown model is the fallback for an unsupported setting, so applying it again is how the
  // user replaces (and clears) that setting. A word equal to the highlighted choice is no change; a typed id
  // is a change unless the model is already pinned to that very id (typing the id of a word's model pins it).
  const unchanged = isCustom ? choice === null && value === current : value === choice;
  const canApply = !busy && valid && (!unchanged || allowReapply);
  const showFormatHint = isCustom && value !== "" && !valid;

  async function onConfirm() {
    setBusy(true);
    setError(null);
    try {
      const { checked } = await setPersonaModelFull(persona, value);
      if (checked === false) onClose(value, { checked: false });
      else onClose(value);
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
