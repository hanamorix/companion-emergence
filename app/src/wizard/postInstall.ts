// The wizard's nell-shortcut step: runs the install and turns the result into the
// install-pane trailer plus an optional durable notice (#408).
//
// macOS wording is byte-identical to what shipped before. Linux and Windows get their own wording.
// Whenever the step reports failure (on macOS since #413), the explanation is also returned as
// `notice`, which the wizard keeps on the final step until the user continues (the install pane is
// gone after 1.2 s).

import { installNellCliSymlink, type InitResult } from "../appConfig";
import { errString } from "../lib/errString";
import { platformLabel, type ClientPlatform } from "../platform";

export interface CliStepResult {
  trailer: string;
  notice: string | null;
}

const VERSION_CMD = "`nell --version`";

function failureResult(trailer: string): CliStepResult {
  // trailer always starts with "\n\n[cli] "
  return { trailer, notice: trailer.replace(/^\n\n\[cli\] /, "") };
}

export async function runCliShortcutStep(
  platform: ClientPlatform,
  install: () => Promise<InitResult> = installNellCliSymlink,
): Promise<CliStepResult> {
  if (platform === "other") {
    return {
      trailer: `\n\n[cli] the nell shortcut is not available on ${platformLabel(platform)}.`,
      notice: null,
    };
  }

  if (platform === "macos") {
    try {
      const cli = await install();
      if (cli.success) {
        return {
          trailer: `\n\n[cli] nell linked to ~/.local/bin \u2014 use ${VERSION_CMD} from Terminal`,
          notice: null,
        };
      }
      // A refusal (exit 73, a user's own nell, #413) must outlast the install pane.
      return failureResult(
        `\n\n[cli] symlink reported exit ${cli.exit_code} \u2014 Terminal access unavailable. Stderr:\n${cli.stderr}`,
      );
    } catch (e) {
      return { trailer: `\n\n[cli] could not install nell symlink: ${errString(e)}`, notice: null };
    }
  }

  try {
    const cli = await install();
    if (cli.success) {
      return {
        trailer:
          platform === "windows"
            ? `\n\n[cli] nell shortcut installed. Open a new Command Prompt or PowerShell window and run ${VERSION_CMD}.`
            : `\n\n[cli] nell shortcut installed in ~/.local/bin. Open a new terminal and run ${VERSION_CMD}. If nell is not found, sign out and back in.`,
        notice: null,
      };
    }
    return failureResult(
      cli.exit_code === 78
        ? `\n\n[cli] nell shortcut not installed.\n${cli.stderr}`
        : `\n\n[cli] nell shortcut not installed (exit ${cli.exit_code}).\n${cli.stderr}`,
    );
  } catch (e) {
    return failureResult(`\n\n[cli] could not install the nell shortcut: ${errString(e)}`);
  }
}
