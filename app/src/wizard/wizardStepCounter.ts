/**
 * Step counter logic for the Wizard — extracted to a thin module so it
 * can be tested without rendering React components.
 *
 * This is the single source of truth for stepNum and totalSteps across
 * all three migrate branches (nellbrain, emergence-kit, companion-emergence)
 * and the fresh install flow.
 *
 * Flow summary:
 *   fresh:             welcome=1, prereq=2,            name=3, user=4, voice=5,         review=6, installing/ready=7  → totalSteps=7
 *   migrate nellbrain: welcome=1, prereq=2, migrate=3, name=4, user=5, voice=6,         review=7, installing/ready=8  → totalSteps=8
 *   migrate kit:       same as nellbrain                                                                               → totalSteps=8
 *   migrate ce:        welcome=1, prereq=2, migrate=3,                                  review=4, installing/ready=5  → totalSteps=5
 *
 * name/user/voice are assigned step 0 for the companion-emergence branch so
 * WizardShell never shows "Step 0 of 5."
 */

import type { MigrateSource } from "../appConfig";

export type WizardMode = "fresh" | "migrate";

export type StepName =
  | "welcome"
  | "prereq"
  | "name"
  | "user"
  | "voice"
  | "migrate"
  | "review"
  | "installing"
  | "ready";

export interface WizardStepInfo {
  totalSteps: number;
  stepNum: Record<StepName, number>;
}

export function computeWizardSteps(
  mode: WizardMode,
  migrateSource: MigrateSource
): WizardStepInfo {
  const isCeMigrate = mode === "migrate" && migrateSource === "companion-emergence";

  const totalSteps = mode === "migrate"
    ? (isCeMigrate ? 5 : 8)
    : 7;

  const stepNum: Record<StepName, number> = isCeMigrate
    ? { welcome: 1, prereq: 2, migrate: 3, name: 0, user: 0, voice: 0, review: 4, installing: 5, ready: 5 }
    : (mode === "migrate"
        ? { welcome: 1, prereq: 2, migrate: 3, name: 4, user: 5, voice: 6, review: 7, installing: 8, ready: 8 }
        : { welcome: 1, prereq: 2, migrate: 0, name: 3, user: 4, voice: 5, review: 6, installing: 7, ready: 7 });

  return { totalSteps, stepNum };
}
