import type { ReactNode } from "react";
import { OptionCard, SectionLabel, WButton, WizardShell } from "../components";
import type { VoiceTemplate } from "../Wizard";

interface Props {
  step: number;
  totalSteps: number;
  template: VoiceTemplate;
  onTemplateChange: (t: VoiceTemplate) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepVoiceTemplate({
  step,
  totalSteps,
  template,
  onTemplateChange,
  onNext,
  onBack,
  avatar,
}: Props) {
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
          <WButton onClick={onNext}>Continue →</WButton>
        </>
      }
    >
      <SectionLabel>Choose a starter</SectionLabel>
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
    </WizardShell>
  );
}
