// C16: exact wizard strings for the nell-shortcut step, per platform and outcome.
// macOS strings are byte-identical to the ones that shipped before #408 (Wizard.tsx 146-150).

import { describe, expect, it, vi } from "vitest";
import type { InitResult } from "../appConfig";
import type { ClientPlatform } from "../platform";
import { runCliShortcutStep } from "./postInstall";

function result(over: Partial<InitResult>): InitResult {
  return { success: true, stdout: "", stderr: "", exit_code: 0, ...over };
}

const V = "`nell --version`";

describe("runCliShortcutStep: macOS is unchanged", () => {
  it("success", async () => {
    const install = vi.fn(async () => result({}));
    const r = await runCliShortcutStep("macos", install);
    expect(install).toHaveBeenCalledTimes(1);
    expect(r).toEqual({
      trailer: `\n\n[cli] nell linked to ~/.local/bin — use ${V} from Terminal`,
      notice: null,
    });
  });

  it("non-success", async () => {
    const r = await runCliShortcutStep("macos", async () =>
      result({ success: false, exit_code: 73, stderr: "Refusing to overwrite" }),
    );
    expect(r).toEqual({
      trailer: "\n\n[cli] symlink reported exit 73 — Terminal access unavailable. Stderr:\nRefusing to overwrite",
      notice: null,
    });
  });

  it("thrown", async () => {
    const r = await runCliShortcutStep("macos", async () => {
      throw new Error("boom");
    });
    expect(r).toEqual({ trailer: "\n\n[cli] could not install nell symlink: boom", notice: null });
  });
});

describe("runCliShortcutStep: Linux and Windows", () => {
  it("Linux success", async () => {
    const install = vi.fn(async () => result({}));
    const r = await runCliShortcutStep("linux", install);
    expect(install).toHaveBeenCalledTimes(1);
    expect(r).toEqual({
      trailer: `\n\n[cli] nell shortcut installed in ~/.local/bin. Open a new terminal and run ${V}. If nell is not found, sign out and back in.`,
      notice: null,
    });
  });

  it("Windows success", async () => {
    const r = await runCliShortcutStep("windows", async () => result({}));
    expect(r).toEqual({
      trailer: `\n\n[cli] nell shortcut installed. Open a new Command Prompt or PowerShell window and run ${V}.`,
      notice: null,
    });
  });

  it.each(["linux", "windows"] as ClientPlatform[])("%s exit 78 keeps the whole stderr text", async (platform) => {
    const stderr = "This AppImage is running without FUSE. Install libfuse2 or the .deb.";
    const r = await runCliShortcutStep(platform, async () =>
      result({ success: false, exit_code: 78, stderr }),
    );
    expect(r.trailer).toBe(`\n\n[cli] nell shortcut not installed.\n${stderr}`);
    expect(r.notice).toBe(`nell shortcut not installed.\n${stderr}`);
  });

  it.each(["linux", "windows"] as ClientPlatform[])("%s other failure includes the exit code", async (platform) => {
    const r = await runCliShortcutStep(platform, async () =>
      result({ success: false, exit_code: 73, stderr: "Refusing to overwrite x" }),
    );
    expect(r.trailer).toBe("\n\n[cli] nell shortcut not installed (exit 73).\nRefusing to overwrite x");
    expect(r.notice).toBe("nell shortcut not installed (exit 73).\nRefusing to overwrite x");
    expect(r.trailer).not.toContain("Terminal access unavailable");
    expect(r.trailer).not.toContain("symlink reported");
  });

  it.each(["linux", "windows"] as ClientPlatform[])("%s thrown error", async (platform) => {
    const r = await runCliShortcutStep(platform, async () => {
      throw new Error("ipc down");
    });
    expect(r.trailer).toBe("\n\n[cli] could not install the nell shortcut: ipc down");
    expect(r.notice).toBe("could not install the nell shortcut: ipc down");
  });
});

describe("runCliShortcutStep: other platforms", () => {
  it("does not call install and says so calmly", async () => {
    const install = vi.fn(async () => result({}));
    const r = await runCliShortcutStep("other", install);
    expect(install).not.toHaveBeenCalled();
    expect(r).toEqual({
      trailer: "\n\n[cli] the nell shortcut is not available on this platform.",
      notice: null,
    });
  });
});
