import type { ReactNode } from "react";
import { OptionCard, SectionLabel, WButton, WizardShell } from "../components";
import type { VoiceTemplate } from "../Wizard";
import { MODEL_ID_MAX_LENGTH, wizardModelIsValid } from "../chatModel";

interface Props {
  step: number;
  totalSteps: number;
  template: VoiceTemplate;
  onTemplateChange: (t: VoiceTemplate) => void;
  chatModel: string;
  onChatModelChange: (m: string) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepVoiceTemplate({
  step,
  totalSteps,
  template,
  onTemplateChange,
  chatModel,
  onChatModelChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  const modelOk = wizardModelIsValid(chatModel);
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Voice template"
      subtitle="voice.md defines how the persona speaks — its register, opinions, and relationship to context."
      avatar={avatar}
      footer={
        <>
          <WButton variant="ghost" onClick={onBack} small>
            ← Back
          </WButton>
          <WButton onClick={onNext} disabled={!modelOk}>Continue →</WButton>
        </>
      }
    >
      <SectionLabel>Choose a starter</SectionLabel>
      <div role="radiogroup" aria-label="Voice template">
        <OptionCard
          selected={template === "nell-example"}
          onClick={() => onTemplateChange("nell-example")}
          title="Original Brain example"
          badge="starter"
          description="Start from the canonical voice.md template. Edit the placeholders to make it your persona's own identity before continuing."
        />
        <OptionCard
          selected={template === "default"}
          onClick={() => onTemplateChange("default")}
          title="Framework default"
          badge="minimal"
          description="Use the framework's built-in DEFAULT_VOICE_TEMPLATE. No file written — you can author voice.md later."
        />
        <OptionCard
          selected={template === "skip"}
          onClick={() => onTemplateChange("skip")}
          title="Skip"
          description="Same as default — no voice.md is written. Provided for scripted setups."
        />
      </div>
      <div
        style={{
          marginTop: 8,
          fontSize: 10.5,
          color: "var(--text-mid)",
          lineHeight: 1.7,
          fontStyle: "italic",
        }}
      >
        {template === "nell-example" &&
          "After install, open ~/Library/Application Support/companion-emergence/personas/<name>/voice.md and replace the example identity content with your own."}
      </div>
      <details style={{ marginTop: 14 }} open={chatModel.trim() !== ""}>
        <summary style={{ fontSize: 11, color: "var(--text-mid)", cursor: "pointer" }}>Advanced</summary>
        <div style={{ marginTop: 8 }}>
          <SectionLabel>Claude model (optional)</SectionLabel>
          <input
            type="text"
            aria-label="Claude model"
            placeholder="sonnet (default)"
            maxLength={MODEL_ID_MAX_LENGTH}
            value={chatModel}
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            onChange={(e) => onChatModelChange(e.target.value)}
            style={{
              width: "100%",
              padding: "6px 8px",
              fontSize: 12,
              fontFamily: "DM Mono, Courier New, monospace",
              color: "var(--text)",
              background: "transparent",
              border: "1px solid var(--hairline)",
              borderRadius: 8,
            }}
          />
          <div style={{ marginTop: 4, fontSize: 10.5, color: "var(--text-mid)", lineHeight: 1.6 }}>
            Leave blank for sonnet, or enter opus, haiku or a full model id such as claude-sonnet-5.
            Checked when first used.
          </div>
          {!modelOk && (
            <div role="alert" style={{ marginTop: 4, fontSize: 10.5, color: "var(--crimson)", lineHeight: 1.5 }}>
              Model IDs start with claude- and use lowercase letters, digits, dots and dashes.
            </div>
          )}
        </div>
      </details>
    </WizardShell>
  );
}
