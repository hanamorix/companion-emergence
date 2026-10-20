import { useState } from "react";
import {
  DEFAULT_INIT_MODEL,
  installSupervisorService,
  runInit,
  runMigrate,
  writeAppConfig,
  type ExistingCePreflight,
  type InitArgs,
  type InitResult,
  type MigrateSource,
} from "../appConfig";
import { getClientPlatform, platformLabel } from "../platform";
import { stopBridgeThen } from "../hooks/useRestartBridge";
import { runCliShortcutStep } from "./postInstall";
import { WizardAvatar } from "./Avatar";
import { StepWelcome } from "./steps/StepWelcome";
import { StepPrerequisites } from "./steps/StepPrerequisites";
import { StepPersonaName } from "./steps/StepPersonaName";
import { StepUserName } from "./steps/StepUserName";
import { StepVoiceTemplate } from "./steps/StepVoiceTemplate";
import { StepMigrate } from "./steps/StepMigrate";
import { StepRecover } from "./steps/StepRecover";
import { StepReview } from "./steps/StepReview";
import { StepInstalling } from "./steps/StepInstalling";
import { StepReady } from "./steps/StepReady";
import { errString } from "../lib/errString";
import { computeWizardSteps } from "./wizardStepCounter";
import { wizardInitModel } from "./chatModel";

export type WizardMode = "fresh" | "migrate";
export type VoiceTemplate = "default" | "nell-example" | "skip";
// MigrateSource is now canonical in appConfig.ts — imported above.
// Re-export for consumers that import from Wizard.tsx directly.
export type { MigrateSource };

export interface WizardState {
  mode: WizardMode;
  personaName: string;
  userName: string;
  userPronouns: string;
  voiceTemplate: VoiceTemplate;
  /** Optional Claude model (alias or full id); blank = the default. */
  chatModel: string;
  migrateFromPath: string;
  migrateSource: MigrateSource;
}

const INITIAL_STATE: WizardState = {
  mode: "fresh",
  personaName: "",
  userName: "",
  userPronouns: "they/them",
  voiceTemplate: "default",
  chatModel: "",
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
  | "recover"
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
  // Explanation from the nell-shortcut step when it did not succeed on Linux/Windows; shown on the
  // final step so it is not lost when the install pane advances (#408).
  const [cliNotice, setCliNotice] = useState<string | null>(null);
  // Preflight result for the existing companion-emergence branch.
  // Held here so StepMigrate, StepReview, and runInstall all share the same object.
  const [preflight, setPreflight] = useState<ExistingCePreflight | null>(null);

  // Step counter and total come from computeWizardSteps — the canonical
  // source of truth for all three migrate branches + fresh flow.
  const isCeMigrate = state.mode === "migrate" && state.migrateSource === "companion-emergence";
  const { totalSteps, stepNum } = computeWizardSteps(state.mode, state.migrateSource);

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
    recover: "migrate",
    review: "review",
    installing: "installing",
    ready: "review",
  };

  // Shared post-install helper — supervisor service + nell CLI symlink.
  // Called from all three install branches so we don't duplicate this logic.
  async function runPostInstall(
    personaName: string,
  ): Promise<{ serviceTrailer: string; cliTrailer: string; cliNotice: string | null }> {
    const platform = getClientPlatform();
    const currentPlatformLabel = platformLabel(platform);
    const supervisorWizardInstall = platform === "macos" || platform === "windows";

    // Plan C — install the persistent supervisor service so it outlives
    // the app from the very first run. macOS (launchd) and Windows (Task
    // Scheduler) both have a real backend arm; on Linux/other we skip
    // quietly so the ready pane doesn't make unsupported platform work
    // look like a failure.
    let serviceTrailer = "";
    if (supervisorWizardInstall) {
      try {
        // #456 (C58 d, review M4): a running bridge is stopped through the stop flow first (the install refuses a
        // running one); the app's launch starts the bridge after the wizard.
        const box: { svc: InitResult | null } = { svc: null };
        const outcome = await stopBridgeThen(
          personaName,
          async () => {
            box.svc = await installSupervisorService(personaName);
          },
          { start: false },
        );
        const svc = box.svc;
        if (!outcome.stopped || svc === null) throw new Error(outcome.error ?? "the brain could not be stopped");
        serviceTrailer = svc.success
          ? `\n\n[service] supervisor installed`
          : `\n\n[service] install reported exit ${svc.exit_code} — supervisor will fall back to the app's lifecycle. Stderr:\n${svc.stderr}`;
      } catch (e) {
        serviceTrailer =
          `\n\n[service] could not install supervisor: ${errString(e)}` +
          " — supervisor will fall back to the app's lifecycle.";
      }
    } else {
      serviceTrailer = `\n\n[service] persistent supervisor install is available from the Connection panel on ${currentPlatformLabel}.`;
    }

    // Put `nell` on the user's PATH so the CLI is reachable from a terminal: a symlink on macOS and
    // Linux (.deb), a small script on Linux (AppImage), nell.cmd plus a PATH entry on Windows. When
    // the step does not succeed on Linux/Windows the explanation is kept as `cliNotice`, shown on
    // the final step until the user continues (#408).
    const cli = await runCliShortcutStep(platform);
    const cliTrailer = cli.trailer;

    return { serviceTrailer, cliTrailer, cliNotice: cli.notice };
  }

  async function runInstall() {
    setCliNotice(null);
    setStep("installing");
    // emergence-kit takes the new auto-importer path: ``nell migrate
    // --source emergence-kit --install-as <name>`` builds the persona
    // dir directly from the kit's JSON files. NellBrain still flows
    // through ``nell init --migrate-from`` because the OG migrator's
    // preflight + lock checks live there.
    const useEmergenceKitMigrator =
      state.mode === "migrate" && state.migrateSource === "emergence-kit";
    const useCompanionEmergenceMigrator =
      state.mode === "migrate" && state.migrateSource === "companion-emergence";

    try {
      let result: InitResult;

      if (useCompanionEmergenceMigrator) {
        // For the existing-CE branch the preflight result carries the persona
        // name — it was already validated by the preflight check in StepMigrate.
        // No follow-up runInit needed: the copied dir already has persona_config.json.
        const personaName = preflight?.persona_name ?? state.personaName;
        result = await runMigrate({
          persona: personaName,
          source: "companion-emergence",
          input_dir: state.migrateFromPath,
          force: false,
        });
        if (!result.success) {
          setInstallResult({
            ok: false,
            output: result.stdout,
            error: result.stderr || `exit ${result.exit_code}`,
          });
          return;
        }
        await writeAppConfig({
          selected_persona: personaName,
          always_on_top: false,
          reduced_motion: false,
        });
        const { serviceTrailer, cliTrailer, cliNotice: notice } = await runPostInstall(personaName);
        setCliNotice(notice);
        setInstallResult({
          ok: true,
          output: result.stdout + serviceTrailer + cliTrailer,
          error: "",
        });
        setTimeout(() => setStep("ready"), 1200);
        return;
      }

      const args: InitArgs = {
        persona: state.personaName,
        user_name: state.userName.trim() || null,
        user_pronouns: state.userPronouns,
        voice_template: state.voiceTemplate,
        migrate_from:
          state.mode === "migrate" && !useEmergenceKitMigrator ? state.migrateFromPath : null,
        force: false,
        model: wizardInitModel(state.chatModel, DEFAULT_INIT_MODEL),
      };

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
      const { serviceTrailer, cliTrailer, cliNotice: notice } = await runPostInstall(state.personaName);
      setCliNotice(notice);
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
      setInstallResult({ ok: false, output: "", error: errString(e) });
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
          // New ordering: in migrate mode, go to the source-picker BEFORE
          // name/user/voice so the CE branch can skip those steps entirely.
          onNext={() => state.mode === "migrate" ? setStep("migrate") : setStep("name")}
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
          // Back from name goes to migrate (if migrate mode) or prereq (fresh).
          onBack={() => state.mode === "migrate" ? setStep("migrate") : setStep("prereq")}
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
          pronouns={state.userPronouns}
          onPronounsChange={(v) => update("userPronouns", v)}
        />
      );
    case "voice":
      return (
        <StepVoiceTemplate
          step={stepNum.voice}
          totalSteps={totalSteps}
          template={state.voiceTemplate}
          onTemplateChange={(t) => update("voiceTemplate", t)}
          chatModel={state.chatModel}
          onChatModelChange={(m) => update("chatModel", m)}
          // voice always proceeds to review (migrate step is now earlier in the flow)
          onNext={() => setStep("review")}
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
          preflight={preflight}
          onPreflightChange={setPreflight}
          // CE branch skips name/user/voice and goes straight to review.
          onNext={() =>
            state.migrateSource === "companion-emergence"
              ? setStep("review")
              : setStep("name")
          }
          onBack={() => setStep("prereq")}
          onRecover={() => setStep("recover")}
          avatar={avatar}
        />
      );
    case "recover":
      return (
        <StepRecover
          persona={
            isCeMigrate && preflight?.persona_name
              ? preflight.persona_name
              : state.personaName
          }
          sourceDir={state.migrateFromPath || null}
          onDone={() => setStep("ready")}
          avatar={avatar}
        />
      );
    case "review":
      return (
        <StepReview
          step={stepNum.review}
          totalSteps={totalSteps}
          state={state}
          preflight={preflight}
          onInstall={runInstall}
          onBack={() => state.mode === "migrate" ? setStep("migrate") : setStep("voice")}
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
          persona={
            isCeMigrate && preflight?.persona_name
              ? preflight.persona_name
              : state.personaName
          }
          onDone={() =>
            onDone(
              isCeMigrate && preflight?.persona_name
                ? preflight.persona_name
                : state.personaName
            )
          }
          avatar={avatar}
          cliNotice={cliNotice}
        />
      );
  }
}
