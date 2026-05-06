import { useState } from "react";
import { runInit, writeAppConfig, type InitArgs } from "../appConfig";
import { WizardAvatar } from "./Avatar";
import { StepWelcome } from "./steps/StepWelcome";
import { StepPersonaName } from "./steps/StepPersonaName";
import { StepUserName } from "./steps/StepUserName";
import { StepLLMSetup } from "./steps/StepLLMSetup";
import { StepVoiceTemplate } from "./steps/StepVoiceTemplate";
import { StepMigrate } from "./steps/StepMigrate";
import { StepReview } from "./steps/StepReview";
import { StepInstalling } from "./steps/StepInstalling";

export type WizardMode = "fresh" | "migrate";
export type VoiceTemplate = "default" | "nell-example" | "skip";

export interface WizardState {
  mode: WizardMode;
  personaName: string;
  userName: string;
  provider: "claude-cli" | "ollama" | "fake";
  voiceTemplate: VoiceTemplate;
  migrateFromPath: string;
}

const INITIAL_STATE: WizardState = {
  mode: "fresh",
  personaName: "",
  userName: "",
  provider: "claude-cli",
  voiceTemplate: "default",
  migrateFromPath: "",
};

type Step =
  | "welcome"
  | "name"
  | "user"
  | "llm"
  | "voice"
  | "migrate"
  | "review"
  | "installing";

interface Props {
  onDone: (persona: string) => void;
}

export function Wizard({ onDone }: Props) {
  const [state, setState] = useState<WizardState>(INITIAL_STATE);
  const [step, setStep] = useState<Step>("welcome");
  const [installResult, setInstallResult] = useState<{
    ok: boolean;
    output: string;
    error: string;
  } | null>(null);

  // Total steps depends on mode — migrate adds the migrate step
  const totalSteps = state.mode === "migrate" ? 7 : 6;
  const stepNum: Record<Step, number> = {
    welcome: 1,
    name: 2,
    user: 3,
    llm: 4,
    voice: 5,
    migrate: 6,
    review: state.mode === "migrate" ? 7 : 6,
    installing: state.mode === "migrate" ? 7 : 6,
  };

  function update<K extends keyof WizardState>(key: K, value: WizardState[K]) {
    setState((s) => ({ ...s, [key]: value }));
  }

  // Step → expression mapping for the avatar
  const stepToExpression: Record<Step, string> = {
    welcome: "welcome",
    name: "name",
    user: "user",
    llm: "llm",
    voice: "voice",
    migrate: "migrate",
    review: "review",
    installing: "installing",
  };

  async function runInstall() {
    setStep("installing");
    const args: InitArgs = {
      persona: state.personaName,
      user_name: state.userName.trim() || null,
      provider: state.provider,
      voice_template: state.voiceTemplate,
      migrate_from: state.mode === "migrate" ? state.migrateFromPath : null,
      force: false,
    };
    try {
      const result = await runInit(args);
      if (result.success) {
        await writeAppConfig({
          selected_persona: state.personaName,
          always_on_top: false,
          reduced_motion: false,
        });
        setInstallResult({ ok: true, output: result.stdout, error: "" });
        // Brief pause so user sees the success state, then route to main app
        setTimeout(() => onDone(state.personaName), 1500);
      } else {
        setInstallResult({
          ok: false,
          output: result.stdout,
          error: result.stderr || `exit ${result.exit_code}`,
        });
      }
    } catch (e) {
      setInstallResult({ ok: false, output: "", error: (e as Error).message });
    }
  }

  const avatarKey = installResult && !installResult.ok
    ? "error"
    : (stepToExpression[step] as keyof typeof stepToExpression);

  const avatar = <WizardAvatar step={avatarKey as Parameters<typeof WizardAvatar>[0]["step"]} />;

  switch (step) {
    case "welcome":
      return (
        <StepWelcome
          step={stepNum.welcome}
          totalSteps={totalSteps}
          mode={state.mode}
          onModeChange={(m) => update("mode", m)}
          onNext={() => setStep("name")}
          avatar={avatar}
        />
      );
    case "name":
      return (
        <StepPersonaName
          step={stepNum.name}
          totalSteps={totalSteps}
          value={state.personaName}
          onChange={(v) => update("personaName", v)}
          onNext={() => setStep("user")}
          onBack={() => setStep("welcome")}
          avatar={avatar}
        />
      );
    case "user":
      return (
        <StepUserName
          step={stepNum.user}
          totalSteps={totalSteps}
          value={state.userName}
          onChange={(v) => update("userName", v)}
          onNext={() => setStep("llm")}
          onBack={() => setStep("name")}
          avatar={avatar}
        />
      );
    case "llm":
      return (
        <StepLLMSetup
          step={stepNum.llm}
          totalSteps={totalSteps}
          provider={state.provider}
          onProviderChange={(p) => update("provider", p)}
          onNext={() => setStep("voice")}
          onBack={() => setStep("user")}
          avatar={avatar}
        />
      );
    case "voice":
      return (
        <StepVoiceTemplate
          step={stepNum.voice}
          totalSteps={totalSteps}
          template={state.voiceTemplate}
          onTemplateChange={(t) => update("voiceTemplate", t)}
          onNext={() => setStep(state.mode === "migrate" ? "migrate" : "review")}
          onBack={() => setStep("llm")}
          avatar={avatar}
        />
      );
    case "migrate":
      return (
        <StepMigrate
          step={stepNum.migrate}
          totalSteps={totalSteps}
          path={state.migrateFromPath}
          onPathChange={(p) => update("migrateFromPath", p)}
          onNext={() => setStep("review")}
          onBack={() => setStep("voice")}
          avatar={avatar}
        />
      );
    case "review":
      return (
        <StepReview
          step={stepNum.review}
          totalSteps={totalSteps}
          state={state}
          onInstall={runInstall}
          onBack={() => setStep(state.mode === "migrate" ? "migrate" : "voice")}
          avatar={avatar}
        />
      );
    case "installing":
      return (
        <StepInstalling
          step={stepNum.installing}
          totalSteps={totalSteps}
          result={installResult}
          onRetry={runInstall}
          onBack={() => {
            setInstallResult(null);
            setStep("review");
          }}
          avatar={avatar}
        />
      );
  }
}
