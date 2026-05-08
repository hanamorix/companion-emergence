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
      title="Welcome to Companion Emergence"
      subtitle="A local-first framework for persistent, emotionally-aware AI companions. Your brain lives on this machine, remembers, dreams between visits, and grows over time."
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
        title="Start fresh"
        badge="recommended"
        description="Create a brand-new persona. Pick a name, a voice, and we'll set up the brain from a blank slate."
      />
      <OptionCard
        selected={mode === "migrate"}
        onClick={() => onModeChange("migrate")}
        title="Migrate from the original framework"
        description="Carry an existing brain over: memories, soul crystallizations, Hebbian edges, and creative DNA. We'll guide you through it."
      />
      <Divider />
      <div style={{ fontSize: 10.5, color: "var(--text-mid)", lineHeight: 1.7 }}>
        Either path goes through the same setup: prerequisites check,
        persona name, your name, voice template, then install. Migration
        adds one extra step to point at your old data.
      </div>
    </WizardShell>
  );
}
