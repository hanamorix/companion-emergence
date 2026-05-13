// Component tests for ConnectionPanel — addresses 2026-05-08 audit P4-2
// (frontend test coverage beyond smoke). Verifies that the status
// banner surfaces real degraded modes and state-poll errors so the
// audit's "user can't tell something's broken" scenario can't regress.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Tauri's invoke is referenced indirectly via installSupervisorService.
// Stub it so the install button can render without trying to talk to
// the real Tauri runtime.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({
    success: true,
    stdout: "service installed",
    stderr: "",
    exit_code: 0,
  })),
}));

vi.mock("../../platform", () => ({
  getClientPlatform: vi.fn(() => "macos"),
  platformLabel: (platform: string) => platform === "macos" ? "macOS" : platform === "windows" ? "Windows" : platform === "linux" ? "Linux" : "this platform",
  supportsMacOnlyInstallActions: (platform: string) => platform === "macos",
}));

import { ConnectionPanel } from "./ConnectionPanel";
import type { PersonaState } from "../../bridge";
import { getClientPlatform } from "../../platform";

function baseState(overrides: Partial<PersonaState> = {}): PersonaState {
  return {
    persona: "test",
    emotions: { love: 9.5 },
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: {
      provider: "claude-cli",
      model: "claude-sonnet-4-6",
      last_heartbeat_at: new Date().toISOString(),
    },
    mode: "live",
    ...overrides,
  };
}

describe("ConnectionPanel — StatusBanner (P3-6 + P4-2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
  });
  afterEach(() => {
    cleanup();
  });

  it("does not render the status banner when mode is live and no error", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("renders a hard-error banner when mode is bridge_down", () => {
    render(
      <ConnectionPanel state={baseState({ mode: "bridge_down" })} persona="test" />,
    );
    const banner = screen.getByRole("alert");
    expect(banner).toBeInTheDocument();
    expect(banner.textContent).toMatch(/bridge offline/i);
  });

  it("renders a hard-error banner when mode is offline", () => {
    render(
      <ConnectionPanel state={baseState({ mode: "offline" })} persona="test" />,
    );
    const banner = screen.getByRole("alert");
    expect(banner.textContent).toMatch(/offline/i);
  });

  it("renders an amber warning banner when mode is provider_down", () => {
    render(
      <ConnectionPanel
        state={baseState({ mode: "provider_down" })}
        persona="test"
      />,
    );
    const banner = screen.getByRole("alert");
    expect(banner.textContent).toMatch(/provider unreachable/i);
  });

  it("surfaces stateError even when mode is live", () => {
    render(
      <ConnectionPanel
        state={baseState()}
        persona="test"
        stateError="ECONNREFUSED 127.0.0.1:55703"
      />,
    );
    const banner = screen.getByRole("alert");
    expect(banner.textContent).toMatch(/state poll failed/i);
    expect(banner.textContent).toMatch(/ECONNREFUSED/);
  });

  it("renders the install supervisor button on macOS", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(
      screen.getByRole("button", { name: /install launchd supervisor/i }),
    ).toBeInTheDocument();
  });

  it("shows calm unsupported notes instead of macOS install buttons on Windows", () => {
    vi.mocked(getClientPlatform).mockReturnValue("windows");
    render(<ConnectionPanel state={baseState()} persona="test" />);

    expect(screen.queryByRole("button", { name: /install launchd supervisor/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /install nell to/i })).not.toBeInTheDocument();
    expect(screen.getAllByText(/macOS-only right now/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/nothing is broken/i)).toBeInTheDocument();
  });

  it("shows calm unsupported notes instead of macOS install buttons on Linux", () => {
    vi.mocked(getClientPlatform).mockReturnValue("linux");
    render(<ConnectionPanel state={baseState()} persona="test" />);

    expect(screen.queryByRole("button", { name: /install launchd supervisor/i })).not.toBeInTheDocument();
    expect(screen.getByText(/on Linux, Companion will use the app-managed supervisor lifecycle/i)).toBeInTheDocument();
  });

  it("renders the privacy row marked accent (local-only)", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByText(/local-only/i)).toBeInTheDocument();
  });
});
