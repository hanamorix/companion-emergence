import type { ReactNode } from "react";
import { OptionCard, SectionLabel, WButton, WizardShell } from "../components";

type Provider = "claude-cli" | "ollama" | "fake";

interface Props {
  step: number;
  totalSteps: number;
  provider: Provider;
  onProviderChange: (p: Provider) => void;
  onNext: () => void;
  onBack: () => void;
  avatar: ReactNode;
}

export function StepLLMSetup({
  step,
  totalSteps,
  provider,
  onProviderChange,
  onNext,
  onBack,
  avatar,
}: Props) {
  return (
    <WizardShell
      step={step}
      totalSteps={totalSteps}
      title="LLM setup"
      subtitle="Choose how the brain talks to a language model."
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
      <SectionLabel>Provider</SectionLabel>
      <OptionCard
        selected={provider === "claude-cli"}
        onClick={() => onProviderChange("claude-cli")}
        title="Claude CLI"
        badge="default"
        description="Shells out to the Claude CLI — uses your existing Claude subscription, no per-token billing."
      />
      <OptionCard
        selected={provider === "ollama"}
        onClick={() => onProviderChange("ollama")}
        title="Ollama (local)"
        badge="local"
        description="Local inference via Ollama — fully private, no API key required. Ollama must be running on localhost."
      />
      <OptionCard
        selected={provider === "fake"}
        onClick={() => onProviderChange("fake")}
        title="Fake (testing)"
        badge="dev"
        description="Deterministic hash-based provider — zero network calls. For development and testing only."
      />
      <div
        style={{
          marginTop: 6,
          fontSize: 10.5,
          color: "var(--text-mid)",
          lineHeight: 1.7,
          fontStyle: "italic",
        }}
      >
        {provider === "claude-cli" &&
          "Requires the claude CLI in your PATH. Install via npm i -g @anthropic-ai/claude-code. The provider shells out per call — no API key needed if you have an active subscription."}
        {provider === "ollama" &&
          "Pull a vision-capable Qwen2.5 / LLaVA model first via ollama pull. Defaults to huihui_ai/qwen2.5-abliterated:7b."}
        {provider === "fake" &&
          "Returns deterministic pseudo-replies seeded by message hash. Useful for tests and offline UI development."}
      </div>
    </WizardShell>
  );
}
