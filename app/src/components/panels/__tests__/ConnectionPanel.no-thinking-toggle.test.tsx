// v0.0.26: extended-reasoning toggle was removed — monologue is always-on.
// This test suite asserts the toggle is gone and that surrounding controls
// (model picker) still render correctly.

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// ── mocks ──────────────────────────────────────────────────────────

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string) => {
    if (cmd === "get_bridge_credentials") {
      return { port: 50000, auth_token: "test-token" };
    }
    return { success: true, stdout: "", stderr: "", exit_code: 0 };
  }),
}));

vi.mock("../../platform", () => ({
  getClientPlatform: vi.fn(() => "macos"),
  platformLabel: (p: string) => p,
  supportsMacOnlyInstallActions: (p: string) => p === "macos",
  detectInstallShape: vi.fn(async () => "native"),
}));

vi.mock("@tauri-apps/plugin-updater", () => ({
  check: vi.fn(async () => null),
  Update: class {},
}));

// ── import ─────────────────────────────────────────────────────────

import { ConnectionPanel } from "../ConnectionPanel";

function makeState() {
  return {
    persona: "nell",
    emotions: { love: 9.5 },
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: {
      provider: "claude-cli",
      model: "sonnet",
      last_heartbeat_at: null,
    },
    mode: "live" as const,
  };
}

// ── tests ──────────────────────────────────────────────────────────

describe("ConnectionPanel — no extended-reasoning toggle (v0.0.26)", () => {
  afterEach(() => {
    cleanup();
  });

  it("does not render an extended-reasoning toggle", () => {
    render(
      <ConnectionPanel
        state={makeState()}
        persona="nell"
      />,
    );
    expect(screen.queryByText(/extended reasoning/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/thinking budget/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("checkbox", { name: /extended reasoning/i }),
    ).not.toBeInTheDocument();
  });

  it("does not render the thinking-before-replies copy", () => {
    render(
      <ConnectionPanel
        state={makeState()}
        persona="nell"
      />,
    );
    expect(
      screen.queryByText(/thinks before she replies/i),
    ).not.toBeInTheDocument();
  });

  it("does not regress the model picker", () => {
    render(
      <ConnectionPanel
        state={makeState()}
        persona="nell"
      />,
    );
    // Model picker shows a "change" link to swap the model.
    expect(screen.getByRole("button", { name: /change/i })).toBeInTheDocument();
  });
});
