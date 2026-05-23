// Boot-phase routing tests for App.tsx.
// Covers the four branches introduced in Bundle 9:
//   1. selected_persona set → skips listPersonas, starts bridge (no picker)
//   2. no selection, 0 personas → wizard
//   3. no selection, 1 persona → auto-selects, writeAppConfig called, no picker
//   4. no selection, ≥2 personas → picker shown

import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// ── Heavy Tauri / Tauri-window deps ──────────────────────────────────────────
vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({ startDragging: vi.fn() }),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => undefined),
}));

// ── appConfig — boot logic under test ────────────────────────────────────────
// vi.mock() is hoisted to the top of the file, so factory-referenced vars
// must be created with vi.hoisted() to be available when the factory runs.
const { readAppConfig, writeAppConfig, listPersonas, ensureBridgeRunning, setAlwaysOnTop } =
  vi.hoisted(() => ({
    readAppConfig: vi.fn(),
    writeAppConfig: vi.fn(async () => undefined),
    listPersonas: vi.fn(),
    ensureBridgeRunning: vi.fn(async () => undefined),
    setAlwaysOnTop: vi.fn(async () => undefined),
  }));

vi.mock("./appConfig", () => ({
  readAppConfig,
  writeAppConfig,
  listPersonas,
  ensureBridgeRunning,
  setAlwaysOnTop,
}));

// ── bridge ────────────────────────────────────────────────────────────────────
vi.mock("./bridge", () => ({
  fetchPersonaState: vi.fn(async () => ({
    persona: "test",
    emotions: {},
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
    mode: "live",
    recovering: false,
    felt_time_recovered: false,
  })),
}));

// ── Heavy UI components that spawn their own effects ─────────────────────────
vi.mock("./components/NellAvatar", () => ({
  NellAvatar: () => <div data-testid="nell-avatar" />,
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

// PersonaPicker renders WizardAvatar which loads expressions — stub the Avatar.
vi.mock("./wizard/Avatar", () => ({
  WizardAvatar: () => null,
}));

// ── Import App after all mocks are in place ───────────────────────────────────
import App from "./App";

function baseConfig(selected_persona: string | null = null) {
  return { selected_persona, always_on_top: false, reduced_motion: false };
}

describe("App boot routing", () => {
  beforeEach(() => {
    readAppConfig.mockReset();
    writeAppConfig.mockReset().mockResolvedValue(undefined);
    listPersonas.mockReset();
    // Default: bridge hangs in-flight so "starting-bridge" phase is observable.
    ensureBridgeRunning.mockReset().mockReturnValue(new Promise(() => undefined));
    setAlwaysOnTop.mockReset().mockResolvedValue(undefined);
  });

  afterEach(cleanup);

  it("selected_persona set → skips listPersonas, shows starting-bridge", async () => {
    readAppConfig.mockResolvedValue(baseConfig("nell"));

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/Starting brain for nell/i)).toBeInTheDocument()
    );
    expect(listPersonas).not.toHaveBeenCalled();
  });

  it("no selection, 0 personas → wizard", async () => {
    readAppConfig.mockResolvedValue(baseConfig(null));
    listPersonas.mockResolvedValue([]);

    render(<App />);

    await waitFor(() =>
      expect(screen.getByTestId("wizard")).toBeInTheDocument()
    );
    expect(writeAppConfig).not.toHaveBeenCalled();
  });

  it("no selection, 1 persona → auto-selects, writeAppConfig called, no picker", async () => {
    readAppConfig.mockResolvedValue(baseConfig(null));
    listPersonas.mockResolvedValue([
      { name: "nell", last_opened_at: "2026-05-22T10:00:00Z", has_memories_db: true },
    ]);

    render(<App />);

    // Bridge is hanging → we land in starting-bridge with the auto-selected persona.
    await waitFor(() =>
      expect(screen.getByText(/Starting brain for nell/i)).toBeInTheDocument()
    );
    expect(writeAppConfig).toHaveBeenCalledWith(
      expect.objectContaining({ selected_persona: "nell" })
    );
    // PersonaPicker ("Which Kindled?") must not appear.
    expect(screen.queryByText(/Which Kindled/i)).not.toBeInTheDocument();
  });

  it("no selection, ≥2 personas → picker shown", async () => {
    readAppConfig.mockResolvedValue(baseConfig(null));
    listPersonas.mockResolvedValue([
      { name: "nell",   last_opened_at: "2026-05-22T10:00:00Z", has_memories_db: true },
      { name: "phoebe", last_opened_at: "2026-05-23T09:00:00Z", has_memories_db: true },
    ]);

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/Which Kindled/i)).toBeInTheDocument()
    );
    expect(screen.getByText(/nell/)).toBeInTheDocument();
    expect(screen.getByText(/phoebe/)).toBeInTheDocument();
    expect(writeAppConfig).not.toHaveBeenCalled();
  });
});
