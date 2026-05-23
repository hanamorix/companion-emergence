// Step counter spec for the Bundle 8 migrate flow reorder.
// Imports computeWizardSteps from wizardStepCounter.ts — the canonical
// source of truth for step numbers across all wizard branches.

import { describe, it, expect, vi } from "vitest";
import { computeWizardSteps } from "./wizardStepCounter";

describe("computeWizardSteps — companion-emergence 5-step short path", () => {
  it("returns totalSteps=5 for companion-emergence migrate", () => {
    const { totalSteps } = computeWizardSteps("migrate", "companion-emergence");
    expect(totalSteps).toBe(5);
  });

  it("name/user/voice are 0 (hidden) for companion-emergence", () => {
    const { stepNum } = computeWizardSteps("migrate", "companion-emergence");
    expect(stepNum.name).toBe(0);
    expect(stepNum.user).toBe(0);
    expect(stepNum.voice).toBe(0);
  });

  it("welcome=1, prereq=2, migrate=3, review=4, installing=5, ready=5", () => {
    const { stepNum } = computeWizardSteps("migrate", "companion-emergence");
    expect(stepNum.welcome).toBe(1);
    expect(stepNum.prereq).toBe(2);
    expect(stepNum.migrate).toBe(3);
    expect(stepNum.review).toBe(4);
    expect(stepNum.installing).toBe(5);
    expect(stepNum.ready).toBe(5);
  });
});

describe("computeWizardSteps — nellbrain migrate 8-step flow", () => {
  it("returns totalSteps=8", () => {
    const { totalSteps } = computeWizardSteps("migrate", "nellbrain");
    expect(totalSteps).toBe(8);
  });

  it("migrate=3 comes BEFORE name=4 (new ordering)", () => {
    const { stepNum } = computeWizardSteps("migrate", "nellbrain");
    expect(stepNum.migrate).toBe(3);
    expect(stepNum.name).toBe(4);
    expect(stepNum.user).toBe(5);
    expect(stepNum.voice).toBe(6);
    expect(stepNum.review).toBe(7);
    expect(stepNum.installing).toBe(8);
    expect(stepNum.ready).toBe(8);
  });
});

describe("computeWizardSteps — emergence-kit migrate 8-step flow", () => {
  it("has same step numbers as nellbrain", () => {
    const kit = computeWizardSteps("migrate", "emergence-kit");
    const nb = computeWizardSteps("migrate", "nellbrain");
    expect(kit.totalSteps).toBe(nb.totalSteps);
    expect(kit.stepNum).toEqual(nb.stepNum);
  });
});

describe("computeWizardSteps — fresh install 7-step flow", () => {
  it("returns totalSteps=7 with migrate=0 (not in flow)", () => {
    const { totalSteps, stepNum } = computeWizardSteps("fresh", "nellbrain");
    expect(totalSteps).toBe(7);
    expect(stepNum.migrate).toBe(0);
    expect(stepNum.name).toBe(3);
    expect(stepNum.review).toBe(6);
    expect(stepNum.installing).toBe(7);
    expect(stepNum.ready).toBe(7);
  });
});

describe("runMigrate — companion-emergence source dispatches correctly", () => {
  it("calls invoke run_migrate with source=companion-emergence", async () => {
    vi.resetModules();
    const invoke = vi.fn(async () => ({
      success: true,
      stdout: "copied 412 memories",
      stderr: "",
      exit_code: 0,
    }));
    vi.doMock("@tauri-apps/api/core", () => ({ invoke }));

    const { runMigrate } = await import("../appConfig");
    await runMigrate({
      persona: "nell",
      source: "companion-emergence",
      input_dir: "/home/user/.local/share/companion-emergence/personas/nell",
      force: false,
    });
    expect(invoke).toHaveBeenCalledWith("run_migrate", {
      args: {
        persona: "nell",
        source: "companion-emergence",
        input_dir: "/home/user/.local/share/companion-emergence/personas/nell",
        force: false,
      },
    });
  });
});

describe("runPreflightExistingCE dispatches preflight_existing_ce", () => {
  it("calls invoke with input_dir and returns preflight shape", async () => {
    vi.resetModules();
    const mockPreflight = {
      ok: true,
      persona_name: "nell",
      imported_user_name: "Hana",
      imported_voice_template: "default",
      memory_count: 412,
      crystallization_count: 38,
      hebbian_edge_count: 1500,
      source_size_bytes: 12_300_000,
      errors: [],
      warnings: [],
    };
    const invoke = vi.fn(async () => mockPreflight);
    vi.doMock("@tauri-apps/api/core", () => ({ invoke }));

    const { runPreflightExistingCE } = await import("../appConfig");
    const result = await runPreflightExistingCE(
      "/home/user/.local/share/companion-emergence/personas/nell"
    );
    expect(invoke).toHaveBeenCalledWith("preflight_existing_ce", {
      input_dir: "/home/user/.local/share/companion-emergence/personas/nell",
    });
    expect(result.ok).toBe(true);
    expect(result.persona_name).toBe("nell");
    expect(result.memory_count).toBe(412);
  });
});
