import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// ── mocks ──────────────────────────────────────────────────────────

// Tauri invoke — returns bridge credentials so getBridgeCredentials works
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

// Mock fetch so we can spy on bridge POSTs without a real server
const mockFetch = vi.fn();
global.fetch = mockFetch;

// ── helpers ────────────────────────────────────────────────────────

import { ConnectionPanel } from "../ConnectionPanel";

type ConnOverride = {
  provider?: string;
  model?: string;
  last_heartbeat_at?: string | null;
  thinking_budget_tokens?: number | null;
};

function makeState(connOverride: ConnOverride = {}) {
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
      thinking_budget_tokens: null,
      ...connOverride,
    },
    mode: "live" as const,
  };
}

// ── tests ──────────────────────────────────────────────────────────

describe("extended reasoning toggle", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true }),
    });
  });

  afterEach(() => {
    cleanup();
  });

  test("toggle renders as off when thinking_budget_tokens is null", () => {
    render(
      <ConnectionPanel
        state={makeState({ thinking_budget_tokens: null })}
        persona="nell"
        companionName="Nell"
      />,
    );
    const toggle = screen.getByRole("checkbox", { name: /extended reasoning/i });
    expect(toggle).not.toBeChecked();
  });

  test("toggle renders as on when thinking_budget_tokens is 8000", () => {
    render(
      <ConnectionPanel
        state={makeState({ thinking_budget_tokens: 8000 })}
        persona="nell"
        companionName="Nell"
      />,
    );
    const toggle = screen.getByRole("checkbox", { name: /extended reasoning/i });
    expect(toggle).toBeChecked();
  });

  test("toggling on calls POST /persona/config/thinking with budget 8000", async () => {
    render(
      <ConnectionPanel
        state={makeState({ thinking_budget_tokens: null })}
        persona="nell"
        companionName="Nell"
      />,
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /extended reasoning/i }));
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/persona/config/thinking"),
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ thinking_budget_tokens: 8000 }),
        }),
      );
    });
  });

  test("toggling off calls POST /persona/config/thinking with null", async () => {
    render(
      <ConnectionPanel
        state={makeState({ thinking_budget_tokens: 8000 })}
        persona="nell"
        companionName="Nell"
      />,
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /extended reasoning/i }));
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/persona/config/thinking"),
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ thinking_budget_tokens: null }),
        }),
      );
    });
  });

  test("optimistically reverts toggle when fetch fails", async () => {
    mockFetch.mockResolvedValue({ ok: false });
    render(
      <ConnectionPanel
        state={makeState({ thinking_budget_tokens: null })}
        persona="nell"
        companionName="Nell"
      />,
    );
    const toggle = screen.getByRole("checkbox", { name: /extended reasoning/i });
    expect(toggle).not.toBeChecked();

    fireEvent.click(toggle);
    // Optimistic: briefly becomes checked
    expect(toggle).toBeChecked();

    await waitFor(() => {
      // Reverted after failure
      expect(toggle).not.toBeChecked();
    });
  });
});
