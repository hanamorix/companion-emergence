import { useState } from "react";
import {
  installNellCliSymlink,
  installSupervisorService,
  runInit,
  runMigrate,
  writeAppConfig,
  type InitArgs,
} from "../appConfig";
import { getClientPlatform, platformLabel, supportsMacOnlyInstallActions } from "../platform";
import { WizardAvatar } from "./Avatar";
import { StepWelcome } from "./steps/StepWelcome";
import { StepPrerequisites } from "./steps/StepPrerequisites";
import { StepPersonaName } from "./steps/StepPersonaName";
import { StepUserName } from "./steps/StepUserName";
import { StepVoiceTemplate } from "./steps/StepVoiceTemplate";
import { StepMigrate } from "./steps/StepMigrate";
import { StepReview } from "./steps/StepReview";
import { StepInstalling } from "./steps/StepInstalling";
import { StepReady } from "./steps/StepReady";

export type WizardMode = "fresh" | "migrate";
export type VoiceTemplate = "default" | "nell-example" | "skip";
export type MigrateSource = "nellbrain" | "emergence-kit";

export interface WizardState {
  mode: WizardMode;
  personaName: string;
  userName: string;
  voiceTemplate: VoiceTemplate;
  migrateFromPath: string;
  migrateSource: MigrateSource;
}

const INITIAL_STATE: WizardState = {
  mode: "fresh",
  personaName: "",
  userName: "",
  voiceTemplate: "default",
  migrateFromPath: "",
  migrateSource: "nellbrain",
};

// Provider is fixed to claude-cli (per project decision 2026-05-07).
// Ollama / fake providers stay valid for tests + scripts but are not
// user-facing GUI choices. PersonaConfig.DEFAULT_PROVIDER picks the
// right value when nell init runs without --provider.
type Step =
  | "welcome"
  | "prereq"
  | "name"
  | "user"
  | "voice"
  | "migrate"
  | "review"
  | "installing"
  | "ready";

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
  const totalSteps = state.mode === "migrate" ? 8 : 7;
  const stepNum: Record<Step, number> = {
    welcome: 1,
    prereq: 2,
    name: 3,
    user: 4,
    voice: 5,
    migrate: 6,
    review: state.mode === "migrate" ? 7 : 6,
    installing: state.mode === "migrate" ? 8 : 7,
    ready: state.mode === "migrate" ? 8 : 7,
  };

  function update<K extends keyof WizardState>(key: K, value: WizardState[K]) {
    setState((s) => ({ ...s, [key]: value }));
  }

  // Step → expression mapping for the avatar
  const stepToExpression: Record<Step, string> = {
    welcome: "welcome",
    prereq: "welcome",
    name: "name",
    user: "user",
    voice: "voice",
    migrate: "migrate",
    review: "review",
    installing: "installing",
    ready: "review",
  };

  async function runInstall() {
    setStep("installing");
    // emergence-kit takes the new auto-importer path: ``nell migrate
    // --source emergence-kit --install-as <name>`` builds the persona
    // dir directly from the kit's JSON files. NellBrain still flows
    // through ``nell init --migrate-from`` because the OG migrator's
    // preflight + lock checks live there.
    const useEmergenceKitMigrator =
      state.mode === "migrate" && state.migrateSource === "emergence-kit";

    const args: InitArgs = {
      persona: state.personaName,
      user_name: state.userName.trim() || null,
      voice_template: state.voiceTemplate,
      migrate_from:
        state.mode === "migrate" && !useEmergenceKitMigrator ? state.migrateFromPath : null,
      force: false,
    };
    try {
      let result;
      if (useEmergenceKitMigrator) {
        // Run the kit migrator first, which creates the persona dir
        // and seeds memories.db / crystallizations.db / personality.json.
        const migrateResult = await runMigrate({
          persona: state.personaName,
          source: "emergence-kit",
          input_dir: state.migrateFromPath,
          force: false,
        });
        if (!migrateResult.success) {
          setInstallResult({
            ok: false,
            output: migrateResult.stdout,
            error: migrateResult.stderr || `exit ${migrateResult.exit_code}`,
          });
          return;
        }
        // Then ``nell init --force`` writes the voice template +
        // persona_config.json on top of the migrated dir without
        // touching memories.db. ``--force`` is needed because the
        // dir now exists.
        result = await runInit({ ...args, force: true });
        if (result.success) {
          // Stitch the migrate output and init output together so
          // the user sees both summaries in the install pane.
          result = {
            ...result,
            stdout: migrateResult.stdout + "\n\n" + result.stdout,
          };
        }
      } else {
        result = await runInit(args);
      }
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
      const platform = getClientPlatform();
      const macInstallActionsSupported = supportsMacOnlyInstallActions(platform);
      const currentPlatformLabel = platformLabel(platform);
      // Plan C — install the launchd LaunchAgent so the supervisor
      // outlives the .app from the very first run. macOS-only; on
      // Windows/Linux we skip quietly so the ready pane doesn't make
      // unsupported platform work look like a failure.
      let serviceTrailer = "";
      if (macInstallActionsSupported) {
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
      } else {
        serviceTrailer = `\n\n[service] persistent service install is macOS-only; on ${currentPlatformLabel}, Companion will use the app-managed supervisor lifecycle.`;
      }
      // Plan C — symlink ~/.local/bin/nell so users can reach the CLI from
      // their Terminal without typing the .app's Resources path. Also
      // macOS-only for now; skip on Windows/Linux with a reassuring note.
      let cliTrailer = "";
      if (macInstallActionsSupported) {
        try {
          const cli = await installNellCliSymlink();
          cliTrailer = cli.success
            ? `\n\n[cli] nell linked to ~/.local/bin — use \`nell --version\` from Terminal`
            : `\n\n[cli] symlink reported exit ${cli.exit_code} — Terminal access unavailable. Stderr:\n${cli.stderr}`;
        } catch (e) {
          cliTrailer = `\n\n[cli] could not install nell symlink: ${(e as Error).message}`;
        }
      } else {
        cliTrailer = `\n\n[cli] bundled CLI shortcut install is macOS-only; no action is needed for normal app use on ${currentPlatformLabel}.`;
      }
      setInstallResult({
        ok: true,
        output: result.stdout + serviceTrailer + cliTrailer,
        error: "",
      });
      // Brief pause so the user reads the install summary, then move
      // to the verification step which polls /persona/state until the
      // brain is fully alive.
      setTimeout(() => setStep("ready"), 1200);
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
          onNext={() => setStep("prereq")}
          avatar={avatar}
        />
      );
    case "prereq":
      return (
        <StepPrerequisites
          step={stepNum.prereq}
          totalSteps={totalSteps}
          onNext={() => setStep("name")}
          onBack={() => setStep("welcome")}
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
          onBack={() => setStep("prereq")}
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
          source={state.migrateSource}
          onSourceChange={(s) => update("migrateSource", s)}
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
    case "ready":
      return (
        <StepReady
          step={stepNum.ready}
          totalSteps={totalSteps}
          persona={state.personaName}
          onDone={() => onDone(state.personaName)}
          avatar={avatar}
        />
      );
  }
}
