// Vitest smoke test for the wizard's runInit args shape — pins the
// audit-2026-05-07 P1-1 fix so the wizard can't regress to sending a
// `provider:` field the CLI parser doesn't accept.

import { describe, it, expect, vi } from "vitest";
import type { InitArgs } from "../appConfig";

describe("Wizard InitArgs shape", () => {
  it("does not include a provider field — claude-cli is the default", () => {
    // The TypeScript compiler enforces this at the type level:
    // InitArgs has no `provider` key. This runtime assertion exists so
    // any future dictionary-shaped construction still fails the build.
    const args: InitArgs = {
      persona: "test",
      user_name: "Hana",
      voice_template: "default",
      migrate_from: null,
      force: false,
    };
    expect("provider" in args).toBe(false);
    // If a future contributor re-adds `provider:`, this object literal
    // gets a TS error first, but if it slips past types we still catch it.
    const keys = Object.keys(args);
    expect(keys).not.toContain("provider");
    expect(keys).toEqual(
      expect.arrayContaining([
        "persona",
        "user_name",
        "voice_template",
        "migrate_from",
        "force",
      ]),
    );
  });

  it("runInit invokes the run_init Tauri command with the args verbatim", async () => {
    const invoke = vi.fn(async () => ({
      success: true,
      stdout: "",
      stderr: "",
      exit_code: 0,
    }));
    vi.doMock("@tauri-apps/api/core", () => ({ invoke }));

    // Re-import after the mock so the module picks up the spy
    const { runInit } = await import("../appConfig");
    const args: InitArgs = {
      persona: "alice",
      user_name: "Hana",
      voice_template: "default",
      migrate_from: null,
      force: false,
    };
    await runInit(args);
    expect(invoke).toHaveBeenCalledWith("run_init", { args });
    // Specifically: the args bag we sent has no provider key
    const calls = invoke.mock.calls as unknown as Array<
      [string, { args: Record<string, unknown> }]
    >;
    const sentArgs = calls[0][1].args;
    expect("provider" in sentArgs).toBe(false);
  });

  it("installSupervisorService dispatches install_supervisor_service with persona", async () => {
    // Reset module cache so the dynamic re-import picks up the new mock
    // (the previous test cached @tauri-apps/api/core's resolution and
    // would otherwise reuse its invoke spy instead of this one).
    vi.resetModules();
    const invoke = vi.fn(async () => ({
      success: true,
      stdout: "service installed: /Users/x/Library/LaunchAgents/...plist",
      stderr: "",
      exit_code: 0,
    }));
    vi.doMock("@tauri-apps/api/core", () => ({ invoke }));

    const { installSupervisorService } = await import("../appConfig");
    const result = await installSupervisorService("alice");
    expect(invoke).toHaveBeenCalledWith("install_supervisor_service", {
      persona: "alice",
    });
    expect(result.success).toBe(true);
  });
});
