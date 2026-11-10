// What App hands the avatar. When nothing is coming in from the bridge or the server (the state poll is failing, or
// there has never been a state), the avatar gets no state and shows the neutral expression; it must not keep the
// last good snapshot and go on showing the last expression.

import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({ startDragging: vi.fn() }),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => undefined),
}));

const { readAppConfig, ensureBridgeRunning } = vi.hoisted(() => ({
  readAppConfig: vi.fn(),
  ensureBridgeRunning: vi.fn(async () => undefined),
}));

vi.mock("./appConfig", () => ({
  readAppConfig,
  writeAppConfig: vi.fn(async () => undefined),
  listPersonas: vi.fn(),
  ensureBridgeRunning,
  setAlwaysOnTop: vi.fn(async () => undefined),
  brainLoginStatus: vi.fn(async () => ({ authorized: true })),
  brainOverlayStatus: vi.fn(async () => ({ active_commit: null as string | null, confirmed: true, undo: "revert" })),
  revertBrain: vi.fn(async () => undefined),
  rollbackBrain: vi.fn(async (_reason: string): Promise<string | null> => null),
  confirmBrainUpdate: vi.fn(async (_commit: string) => undefined),
  nellbrainHomePath: vi.fn(async (): Promise<string | null> => null),
}));

import type { PersonaState } from "./bridge";

const { fetchPersonaState } = vi.hoisted(() => ({
  fetchPersonaState: vi.fn(),
}));

vi.mock("./bridge", () => ({
  fetchPersonaState,
  approvePendingWrite: vi.fn(async () => ({ ok: true })),
  declinePendingWrite: vi.fn(async () => ({ ok: true })),
  bridgeStatus: vi.fn(async () => ({ alive: "gone" })),
}));

// The avatar is a probe: it reports whether App gave it a state, and which emotions the state held.
vi.mock("./components/NellAvatar", () => ({
  NellAvatar: ({ state }: { state: PersonaState | null }) => (
    <div
      data-testid="nell-avatar"
      data-state={state === null ? "none" : "snapshot"}
      data-emotions={state === null ? "" : Object.keys(state.emotions).join(",")}
    />
  ),
}));

vi.mock("./components/ChatPanel", () => ({
  ChatPanel: () => <div data-testid="chat-panel" />,
}));

vi.mock("./components/LeftPanel", () => ({
  LeftPanel: () => <div data-testid="left-panel" />,
}));

vi.mock("./useSoulFlash", () => ({
  useSoulFlash: () => false,
}));

vi.mock("./wizard/Wizard", () => ({
  Wizard: () => <div data-testid="wizard" />,
}));

vi.mock("./wizard/Avatar", () => ({
  WizardAvatar: () => null,
}));

// A failing state poll is the case under test; the watchdog would start fake restarts of a bridge that is not there.
vi.mock("./hooks/useBridgeWatchdog", () => ({
  useBridgeWatchdog: () => undefined,
}));

vi.mock("./bridgeVersionCheck", () => ({
  ensureBridgeCurrent: vi.fn(async () => "ok"),
  _resetForTests: vi.fn(),
}));

import App from "./App";

function snapshot(): PersonaState {
  return {
    persona: "nell",
    emotions: { joy: 5 },
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
    mode: "live",
    recovering: false,
    felt_time_recovered: false,
  };
}

const POLL_MS = 5000;

describe("App avatar input", () => {
  beforeEach(() => {
    readAppConfig.mockReset().mockResolvedValue({ selected_persona: "nell", always_on_top: false, reduced_motion: false });
    ensureBridgeRunning.mockReset().mockResolvedValue(undefined);
    fetchPersonaState.mockReset();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("gives the avatar the state while the polls succeed", async () => {
    fetchPersonaState.mockResolvedValue(snapshot());
    render(<App />);

    await waitFor(() => expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-state", "snapshot"));
    expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-emotions", "joy");
  });

  it("gives the avatar no state when there has never been one (the first poll fails)", async () => {
    fetchPersonaState.mockRejectedValue(new Error("bridge down"));
    render(<App />);

    await waitFor(() => expect(fetchPersonaState).toHaveBeenCalled());
    expect(await screen.findByTestId("nell-avatar")).toHaveAttribute("data-state", "none");
  });

  it("gives the avatar no state, not the last good snapshot, once a later poll fails", async () => {
    fetchPersonaState
      .mockResolvedValueOnce(snapshot()) // poll 1: good
      .mockRejectedValue(new Error("bridge down")); // poll 2 and after: down
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-state", "snapshot"));

    await vi.advanceTimersByTimeAsync(POLL_MS); // poll 2
    await waitFor(() => expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-state", "none"));
    expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-emotions", "");
  });

  it("gives the avatar the state again when the next poll succeeds", async () => {
    fetchPersonaState
      .mockResolvedValueOnce(snapshot()) // poll 1: good
      .mockRejectedValueOnce(new Error("bridge down")) // poll 2: down
      .mockResolvedValue(snapshot()); // poll 3 and after: back
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-state", "snapshot"));

    await vi.advanceTimersByTimeAsync(POLL_MS); // poll 2
    await waitFor(() => expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-state", "none"));

    await vi.advanceTimersByTimeAsync(POLL_MS); // poll 3
    await waitFor(() => expect(screen.getByTestId("nell-avatar")).toHaveAttribute("data-state", "snapshot"));
  });
});
