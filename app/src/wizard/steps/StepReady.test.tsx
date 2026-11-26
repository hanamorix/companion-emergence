// Component tests for StepReady — addresses 2026-05-08 audit P2-11
// (the wizard previously gated readiness on emotionCount > 0, which
// stalls fresh personas where the first heartbeat hasn't fired yet)
// and contributes to P4-2 frontend coverage.

import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Mocks have to be declared before the component import so the
// dynamic resolution picks them up. ensureBridgeRunning + fetchPersonaState
// are the two lifecycle calls StepReady drives.
const ensureBridgeRunning = vi.fn();
const fetchPersonaState = vi.fn();

vi.mock("../../appConfig", () => ({
  ensureBridgeRunning: (...args: unknown[]) => ensureBridgeRunning(...args),
}));
vi.mock("../../bridge", () => ({
  fetchPersonaState: (...args: unknown[]) => fetchPersonaState(...args),
}));
// WizardShell pulls in nothing else dangerous — no mock needed.

import { StepReady } from "./StepReady";

function freshState(emotions: Record<string, number> = {}) {
  return {
    persona: "test",
    emotions,
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
    mode: "live" as const,
  };
}

describe("StepReady (P2-11 + P4-2)", () => {
  beforeEach(() => {
    ensureBridgeRunning.mockReset();
    fetchPersonaState.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("auto-completes when /persona/state is reachable, even with empty emotions", async () => {
    ensureBridgeRunning.mockResolvedValue(undefined);
    fetchPersonaState.mockResolvedValue(freshState({}));
    const onDone = vi.fn();

    render(
      <StepReady
        step={1}
        totalSteps={1}
        persona="test"
        onDone={onDone}
        avatar={<div data-testid="avatar" />}
      />,
    );

    // Bridge stage flips from running to ok almost immediately.
    await waitFor(() =>
      expect(screen.getByText(/Bringing the brain online/)).toBeInTheDocument(),
    );

    await waitFor(() => expect(onDone).toHaveBeenCalled(), { timeout: 5000 });
  });

  it("shows the warmup detail line when emotions are empty", async () => {
    ensureBridgeRunning.mockResolvedValue(undefined);
    fetchPersonaState.mockResolvedValue(freshState({}));

    render(
      <StepReady
        step={1}
        totalSteps={1}
        persona="test"
        onDone={() => undefined}
        avatar={<div />}
      />,
    );

    await waitFor(() =>
      expect(
        screen.getByText(/heartbeat will warm this up shortly/i),
      ).toBeInTheDocument(),
    );
  });

  it("surfaces an error fallthrough when ensureBridgeRunning rejects", async () => {
    ensureBridgeRunning.mockRejectedValue(new Error("supervisor_start_timeout"));
    fetchPersonaState.mockResolvedValue(freshState());

    const onDone = vi.fn();
    render(
      <StepReady
        step={1}
        totalSteps={1}
        persona="test"
        onDone={onDone}
        avatar={<div />}
      />,
    );

    await waitFor(() =>
      expect(screen.getByText(/Open the app anyway/i)).toBeInTheDocument(),
    );
    expect(onDone).not.toHaveBeenCalled();
  });

  it("renders the populated emotion preview when state has emotions", async () => {
    ensureBridgeRunning.mockResolvedValue(undefined);
    fetchPersonaState.mockResolvedValue(freshState({ love: 9.5, grief: 7.2 }));

    render(
      <StepReady
        step={1}
        totalSteps={1}
        persona="test"
        onDone={() => undefined}
        avatar={<div />}
      />,
    );

    await waitFor(() => expect(screen.getByText(/love/i)).toBeInTheDocument());
  });
});

// ── durable shortcut notice (#408, C29 a2) ─────────────────────────────

describe("StepReady: durable nell shortcut notice", () => {
  const NOTICE =
    "nell shortcut not installed. This AppImage is running without FUSE. Install libfuse2 or use the .deb.";

  beforeEach(() => {
    ensureBridgeRunning.mockReset();
    fetchPersonaState.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("keeps the notice on screen, does not auto-advance, and Continue leaves", async () => {
    vi.useFakeTimers();
    ensureBridgeRunning.mockResolvedValue(undefined);
    fetchPersonaState.mockResolvedValue(freshState({}));
    const onDone = vi.fn();
    render(
      <StepReady
        step={1}
        totalSteps={1}
        persona="test"
        onDone={onDone}
        avatar={<div />}
        cliNotice={NOTICE}
      />,
    );
    await vi.advanceTimersByTimeAsync(6000);
    expect(screen.getByRole("status")).toHaveTextContent(NOTICE);
    expect(onDone).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("Continue is visible from the start", () => {
    ensureBridgeRunning.mockReturnValue(new Promise(() => undefined));
    fetchPersonaState.mockResolvedValue(freshState({}));
    render(
      <StepReady
        step={1}
        totalSteps={1}
        persona="test"
        onDone={() => undefined}
        avatar={<div />}
        cliNotice={NOTICE}
      />,
    );
    expect(screen.getByRole("button", { name: /continue/i })).toBeInTheDocument();
  });

  it("without a notice it still auto-advances and shows no status block or Continue button", async () => {
    vi.useFakeTimers();
    ensureBridgeRunning.mockResolvedValue(undefined);
    fetchPersonaState.mockResolvedValue(freshState({}));
    const onDone = vi.fn();
    render(
      <StepReady step={1} totalSteps={1} persona="test" onDone={onDone} avatar={<div />} cliNotice={null} />,
    );
    await vi.advanceTimersByTimeAsync(3000);
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^continue/i })).not.toBeInTheDocument();
  });
});
