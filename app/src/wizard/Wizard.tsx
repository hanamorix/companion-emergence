import { useState } from "react";
import {
  installSupervisorService,
  runInit,
  writeAppConfig,
  type InitArgs,
} from "../appConfig";
import { WizardAvatar } from "./Avatar";
import { StepWelcome } from "./steps/StepWelcome";
import { StepPersonaName } from "./steps/StepPersonaName";
import { StepUserName } from "./steps/StepUserName";
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
  voiceTemplate: VoiceTemplate;
  migrateFromPath: string;
}

const INITIAL_STATE: WizardState = {
  mode: "fresh",
  personaName: "",
  userName: "",
  voiceTemplate: "default",
  migrateFromPath: "",
};

// Provider is fixed to claude-cli (per project decision 2026-05-07).
// Ollama / fake providers stay valid for tests + scripts but are not
// user-facing GUI choices. PersonaConfig.DEFAULT_PROVIDER picks the
// right value when nell init runs without --provider.
type Step =
  | "welcome"
  | "name"
  | "user"
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
  const totalSteps = state.mode === "migrate" ? 6 : 5;
  const stepNum: Record<Step, number> = {
    welcome: 1,
    name: 2,
    user: 3,
    voice: 4,
    migrate: 5,
    review: state.mode === "migrate" ? 6 : 5,
    installing: state.mode === "migrate" ? 6 : 5,
  };

  function update<K extends keyof WizardState>(key: K, value: WizardState[K]) {
    setState((s) => ({ ...s, [key]: value }));
  }

  // Step → expression mapping for the avatar
  const stepToExpression: Record<Step, string> = {
    welcome: "welcome",
    name: "name",
    user: "user",
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
      voice_template: state.voiceTemplate,
      migrate_from: state.mode === "migrate" ? state.migrateFromPath : null,
      force: false,
    };
    try {
      const result = await runInit(args);
      if (!result.success) {
        setInstallResult({
          ok: false,
          output: result.stdout,
          error: result.stderr || `exit ${result.exit_code}`,
        });
        return;
      }
      await writeAppConfig({
        selected_persona: state.personaName,
        always_on_top: false,
        reduced_motion: false,
      });
      // Plan C — install the launchd LaunchAgent so the supervisor
      // outlives the .app from the very first run. Best-effort: a
      // failure here doesn't block the wizard (the legacy Tauri-spawn
      // path still works), but we surface the stderr in the success
      // pane so the user can fix it from the connection panel later.
      let serviceTrailer = "";
      try {
        const svc = await installSupervisorService(state.personaName);
        serviceTrailer = svc.success
          ? `\n\n[service] launchd agent installed`
          : `\n\n[service] install reported exit ${svc.exit_code} — supervisor will fall back to the .app's lifecycle. Stderr:\n${svc.stderr}`;
      } catch (e) {
        serviceTrailer =
          `\n\n[service] could not install launchd agent: ${(e as Error).message}` +
          " — supervisor will fall back to the .app's lifecycle.";
      }
      setInstallResult({
        ok: true,
        output: result.stdout + serviceTrailer,
        error: "",
      });
      // Brief pause so user sees the success state, then route to main app
      setTimeout(() => onDone(state.personaName), 1500);
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
          onNext={() => setStep("voice")}
          onBack={() => setStep("name")}
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
          onBack={() => setStep("user")}
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
