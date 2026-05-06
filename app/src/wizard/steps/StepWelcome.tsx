import type { ReactNode } from "react";
import { Divider, OptionCard, SectionLabel, WButton, WizardShell } from "../components";
import type { WizardMode } from "../Wizard";

interface Props {
  step: number;
  totalSteps: number;
  mode: WizardMode;
  onModeChange: (mode: WizardMode) => void;
  onNext: () => void;
  avatar: ReactNode;
}

export function StepWelcome({ step, totalSteps, mode, onModeChange, onNext, avatar }: Props) {
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="Welcome"
      subtitle="Set up a new brain and persona, or migrate an existing one from the original framework."
      avatar={avatar}
      footer={
        <>
          <span
            style={{
              fontSize: 10.5,
              color: "var(--text-mute)",
              fontFamily: "var(--font-disp)",
              fontStyle: "italic",
            }}
          >
            companion-emergence v0.1
          </span>
          <WButton onClick={onNext}>Continue →</WButton>
        </>
      }
    >
      <SectionLabel>What would you like to do?</SectionLabel>
      <OptionCard
        selected={mode === "fresh"}
        onClick={() => onModeChange("fresh")}
        title="Fresh brain"
        badge="recommended"
        description="Create a new persona from scratch — choose a name, voice, and LLM provider."
      />
      <OptionCard
        selected={mode === "migrate"}
        onClick={() => onModeChange("migrate")}
        title="Migrate from Original Brain"
        description="Port an existing brain's memories, soul, Hebbian edges, and creative DNA to the new framework."
      />
      <Divider />
      <div style={{ fontSize: 10.5, color: "var(--text-mid)", lineHeight: 1.7 }}>
        Both paths let you configure voice, provider, and persona name before finishing. Migration adds a data-import step.
      </div>
    </WizardShell>
  );
}
