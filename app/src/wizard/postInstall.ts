// The wizard's nell-shortcut step: runs the install and turns the result into the
// install-pane trailer plus an optional durable notice (#408).
// TESTS-FIRST STUB: returns an empty trailer so the pinned strings in postInstall.test.ts fail.

import type { InitResult } from "../appConfig";
import type { ClientPlatform } from "../platform";

export interface CliStepResult {
  trailer: string;
  notice: string | null;
}

export async function runCliShortcutStep(
  _platform: ClientPlatform,
  _install: () => Promise<InitResult>,
): Promise<CliStepResult> {
  return { trailer: "", notice: null };
}
