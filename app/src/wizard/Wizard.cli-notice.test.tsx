// C29(a): the REAL Wizard keeps the nell-shortcut explanation on screen after the install step.
// A new file on purpose: Wizard.test.tsx imports the real ../appConfig after vi.doMock, and a
// hoisted vi.mock here would break it. The input steps are stubbed to one-button components so the
// real Wizard state machine, runInstall, and the real StepReady run.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

const NO_FUSE_TEXT =
  "This AppImage is running without FUSE (in extract-and-run mode), so the nell shortcut was not installed. A shortcut would re-extract the whole app on every nell command and delete the running app's files when the command ends. To get the shortcut, install FUSE for your distribution (on Debian and Ubuntu the package is libfuse2, called libfuse2t64 on Ubuntu 24.04), start the AppImage normally, and try again. Or install the .deb package instead, which does not need FUSE.";

const installNellCliSymlink = vi.fn();
// #456 (C58 d): the install runs inside the stop flow; here the flow just records itself and runs the install.
const { platformBox, order } = vi.hoisted(() => ({
  platformBox: { value: "linux" as "linux" | "macos" },
  order: [] as string[],
}));
vi.mock("../hooks/useRestartBridge", () => ({
  stopBridgeThen: vi.fn(async (_persona: string, action: () => Promise<void>, opts: { start: boolean }) => {
    order.push(`stop (start after: ${opts.start})`);
    await action();
    return { stopped: true, error: null };
  }),
}));

vi.mock("../appConfig", async (importOriginal) => {
  const real = await importOriginal<typeof import("../appConfig")>();
  return {
    ...real,
    runInit: vi.fn(async () => ({ success: true, stdout: "init ok", stderr: "", exit_code: 0 })),
    runMigrate: vi.fn(async () => ({ success: true, stdout: "", stderr: "", exit_code: 0 })),
    writeAppConfig: vi.fn(async () => undefined),
    installSupervisorService: vi.fn(async () => {
      order.push("install");
      return { success: true, stdout: "", stderr: "", exit_code: 0 };
    }),
    installNellCliSymlink: (...a: unknown[]) => installNellCliSymlink(...a),
    ensureBridgeRunning: vi.fn(async () => undefined),
  };
});
vi.mock("../bridge", () => ({
  fetchPersonaState: vi.fn(async () => ({
    persona: "x",
    emotions: {},
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
    mode: "live",
  })),
}));
vi.mock("../platform", async (importOriginal) => {
  const real = await importOriginal<typeof import("../platform")>();
  return { ...real, getClientPlatform: () => platformBox.value };
});
vi.mock("./Avatar", () => ({ WizardAvatar: () => <div data-testid="avatar" /> }));
vi.mock("./steps/StepWelcome", () => ({
  StepWelcome: (p: { onNext: () => void }) => <button onClick={p.onNext}>welcome-next</button>,
}));
vi.mock("./steps/StepPrerequisites", () => ({
  StepPrerequisites: (p: { onNext: () => void }) => <button onClick={p.onNext}>prereq-next</button>,
}));
vi.mock("./steps/StepPersonaName", () => ({
  StepPersonaName: (p: { onNext: () => void }) => <button onClick={p.onNext}>name-next</button>,
}));
vi.mock("./steps/StepUserName", () => ({
  StepUserName: (p: { onNext: () => void }) => <button onClick={p.onNext}>user-next</button>,
}));
vi.mock("./steps/StepVoiceTemplate", () => ({
  StepVoiceTemplate: (p: { onNext: () => void }) => <button onClick={p.onNext}>voice-next</button>,
}));
vi.mock("./steps/StepMigrate", () => ({ StepMigrate: () => <div /> }));
vi.mock("./steps/StepRecover", () => ({ StepRecover: () => <div /> }));
vi.mock("./steps/StepReview", () => ({
  StepReview: (p: { onInstall: () => void }) => <button onClick={p.onInstall}>review-install</button>,
}));
vi.mock("./steps/StepInstalling", () => ({ StepInstalling: () => <div>installing</div> }));

import { Wizard } from "./Wizard";

async function driveToInstall(onDone: () => void) {
  render(<Wizard onDone={onDone} />);
  for (const label of ["welcome-next", "prereq-next", "name-next", "user-next", "voice-next", "review-install"]) {
    fireEvent.click(screen.getByText(label));
  }
  // let runInstall's promises settle, then pass the 1200 ms install-summary pause
  await vi.advanceTimersByTimeAsync(1500);
}

describe("Wizard: the nell-shortcut explanation survives the install step", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    installNellCliSymlink.mockReset();
    platformBox.value = "linux";
    order.length = 0;
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("no-FUSE AppImage: the full message stays on the final step and the wizard waits for Continue", async () => {
    installNellCliSymlink.mockResolvedValue({ success: false, stdout: "", stderr: NO_FUSE_TEXT, exit_code: 78 });
    const onDone = vi.fn();
    await driveToInstall(onDone);
    await vi.advanceTimersByTimeAsync(8000); // well past the state-poll auto-advance
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(document.body.textContent).toContain(NO_FUSE_TEXT);
    expect(onDone).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("control: a successful shortcut keeps the old auto-advance and shows no notice", async () => {
    installNellCliSymlink.mockResolvedValue({ success: true, stdout: "ok", stderr: "", exit_code: 0 });
    const onDone = vi.fn();
    await driveToInstall(onDone);
    await vi.advanceTimersByTimeAsync(8000);
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("(#456 C58 d, review M4) macOS: the supervisor install runs on a bridge stopped through the stop flow", async () => {
    platformBox.value = "macos";
    installNellCliSymlink.mockResolvedValue({ success: true, stdout: "ok", stderr: "", exit_code: 0 });
    await driveToInstall(vi.fn());
    expect(order).toEqual(["stop (start after: false)", "install"]);
  });
});
