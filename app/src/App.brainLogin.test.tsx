// Task 4 — brain-login banner wiring on App.tsx.
// THE LOAD-BEARING INVARIANT: the main chat UI must render regardless of
// brain-login auth state. This banner is an offer, never a gate.

import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement Element.scrollTo; ChatPanel's auto-scroll effect
// needs it (it is NOT mocked away in this file — see note below).
beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

// ── Heavy Tauri / Tauri-window deps ──────────────────────────────────────────
vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({ startDragging: vi.fn() }),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => undefined),
}));

// ── appConfig — boot logic + brain-login status under test ──────────────────
const {
  readAppConfig,
  writeAppConfig,
  listPersonas,
  ensureBridgeRunning,
  setAlwaysOnTop,
  brainLoginStatus,
} = vi.hoisted(() => ({
  readAppConfig: vi.fn(),
  writeAppConfig: vi.fn(async () => undefined),
  listPersonas: vi.fn(),
  ensureBridgeRunning: vi.fn(async () => undefined),
  setAlwaysOnTop: vi.fn(async () => undefined),
  brainLoginStatus: vi.fn(async () => ({ authorized: true })),
}));

vi.mock("./appConfig", () => ({
  readAppConfig,
  writeAppConfig,
  listPersonas,
  ensureBridgeRunning,
  setAlwaysOnTop,
  brainLoginStatus,
}));

// ── bridge ────────────────────────────────────────────────────────────────────
// NOTE: ChatPanel is intentionally NOT mocked in this file (unlike
// App.test.tsx) — the invariant this test proves requires the real chat
// UI's data-testid. That means this mock must also satisfy ChatPanel's
// own imports from "./bridge" (newSession, fetchActiveSession, etc), not
// just App's (fetchPersonaState, approve/declinePendingWrite).
import type { PersonaState } from "./bridge";

const {
  fetchPersonaState,
  approvePendingWrite,
  declinePendingWrite,
  newSession,
  fetchActiveSession,
  fetchChatHistory,
  closeSession,
  snapshotSession,
  uploadImage,
  getBridgeCredentials,
  acceptVoiceEdit,
  rejectVoiceEdit,
} = vi.hoisted(() => {
  const baseState = (): PersonaState => ({
    persona: "test",
    emotions: {},
    body: null,
    interior: { dream: null, research: null, heartbeat: null, reflex: null },
    soul_highlight: null,
    connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
    mode: "live",
    recovering: false,
    felt_time_recovered: false,
  });
  return {
    fetchPersonaState: vi.fn(async (): Promise<PersonaState> => baseState()),
    approvePendingWrite: vi.fn(async () => ({ ok: true })),
    declinePendingWrite: vi.fn(async () => ({ ok: true })),
    newSession: vi.fn(async () => "test-session-id"),
    fetchActiveSession: vi.fn(async () => null),
    fetchChatHistory: vi.fn(async () => ({ messages: [], next_before_turn: null })),
    closeSession: vi.fn(async () => undefined),
    snapshotSession: vi.fn(async () => ({ closed: false, errors: 0 })),
    uploadImage: vi.fn(async () => ({ sha: "deadbeef" })),
    getBridgeCredentials: vi.fn(async () => ({
      url: "http://127.0.0.1:50000",
      port: 50000,
      authToken: "test-token",
    })),
    acceptVoiceEdit: vi.fn(async () => ({ ok: true })),
    rejectVoiceEdit: vi.fn(async () => ({ ok: true })),
  };
});

vi.mock("./bridge", () => ({
  fetchPersonaState,
  approvePendingWrite,
  declinePendingWrite,
  newSession,
  fetchActiveSession,
  fetchChatHistory,
  closeSession,
  snapshotSession,
  uploadImage,
  getBridgeCredentials,
  acceptVoiceEdit,
  rejectVoiceEdit,
}));

// ChatPanel opens a real WebSocket via bridgeEvents unless stubbed.
vi.mock("./bridgeEvents", () => ({
  subscribeToBridgeEvents: vi.fn(() => ({
    subscribe: () => () => undefined,
    close: () => undefined,
  })),
}));

vi.mock("./streamChat", () => ({
  streamChat: vi.fn(async () => () => undefined),
}));

// ChatPanel's new glass header (Phase 4) resolves an avatar thumb via
// expressions.ts, whose import.meta.glob eagerly pulls every expression
// PNG at module-load time. Under this worktree's checkout, Vite's
// fs.allow root-detection (searchForWorkspaceRoot) stops at the worktree
// root because .git there is a file, not a directory — so it never
// widens to include the sibling `expressions/` dir, and the eager glob's
// fetch is denied. Stub the module so this behavior-focused test doesn't
// depend on that asset pipeline at all (this test doesn't render real
// ChatPanel/NellAvatar art either way).
vi.mock("./expressions", () => ({
  resolveFrameUrl: () => "",
}));

// ── Heavy UI components that spawn their own effects ─────────────────────────
vi.mock("./components/NellAvatar", () => ({
  NellAvatar: () => <div data-testid="nell-avatar" />,
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

// ── bridgeVersionCheck ────────────────────────────────────────────────────────
const { ensureBridgeCurrent } = vi.hoisted(() => ({
  ensureBridgeCurrent: vi.fn(
    async (): Promise<"ok" | "restarted" | "version_mismatch_unresolved" | "skipped"> => "ok",
  ),
}));

vi.mock("./bridgeVersionCheck", () => ({
  ensureBridgeCurrent,
  _resetForTests: vi.fn(),
}));

// ── Import App after all mocks are in place ───────────────────────────────────
import App from "./App";

function baseConfig(selected_persona: string | null = "nell") {
  return { selected_persona, always_on_top: false, reduced_motion: false };
}

describe("App brain-login banner", () => {
  beforeEach(() => {
    readAppConfig.mockReset().mockResolvedValue(baseConfig("nell"));
    writeAppConfig.mockReset().mockResolvedValue(undefined);
    listPersonas.mockReset();
    ensureBridgeRunning.mockReset().mockResolvedValue(undefined);
    ensureBridgeCurrent.mockReset().mockResolvedValue("ok");
    setAlwaysOnTop.mockReset().mockResolvedValue(undefined);
    brainLoginStatus.mockReset().mockResolvedValue({ authorized: true });
    fetchPersonaState.mockReset().mockResolvedValue({
      persona: "nell",
      emotions: {},
      body: null,
      interior: { dream: null, research: null, heartbeat: null, reflex: null },
      soul_highlight: null,
      connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
      mode: "live",
      recovering: false,
      felt_time_recovered: false,
    });
  });

  afterEach(cleanup);

  it("renders the chat/main UI even when brain login is unauthorized, alongside a dismissible offer", async () => {
    brainLoginStatus.mockResolvedValue({ authorized: false });

    render(<App />);

    // Invariant: the real chat UI is present regardless of auth state.
    await waitFor(() => expect(screen.getByTestId("chat-messages")).toBeInTheDocument());

    // The offer is present too — but non-blocking (it never replaced chat).
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /authorize/i })).toBeInTheDocument(),
    );
  });

  it("does not show the authorize offer when already authorized", async () => {
    brainLoginStatus.mockResolvedValue({ authorized: true });

    render(<App />);

    await waitFor(() => expect(screen.getByTestId("chat-messages")).toBeInTheDocument());
    await waitFor(() => expect(brainLoginStatus).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /authorize/i })).not.toBeInTheDocument();
  });

  it("dismissing the banner hides it without touching chat", async () => {
    const { fireEvent } = await import("@testing-library/react");
    brainLoginStatus.mockResolvedValue({ authorized: false });

    render(<App />);

    await waitFor(() => expect(screen.getByTestId("chat-messages")).toBeInTheDocument());
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /not now/i })).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: /not now/i }));

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /authorize/i })).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("chat-messages")).toBeInTheDocument();
  });

  it("swallows brainLoginStatus failures — treated as no banner, chat still renders", async () => {
    brainLoginStatus.mockRejectedValue(new Error("boom"));

    render(<App />);

    await waitFor(() => expect(screen.getByTestId("chat-messages")).toBeInTheDocument());
    await waitFor(() => expect(brainLoginStatus).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /authorize/i })).not.toBeInTheDocument();
  });

  it("chat header pill falls back to bridge_down when the state poll has failed, matching GlobalStatusDot", async () => {
    // fetchPersonaState rejecting is what sets App's `stateError` — the
    // same signal GlobalStatusDot already renders crimson for. The chat
    // header pill must not default to green "live" in this state; it
    // doubles as a health readout and should agree with the status dot.
    fetchPersonaState.mockRejectedValue(new Error("network down"));

    render(<App />);

    await waitFor(() => expect(screen.getByTestId("chat-messages")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("● bridge down")).toBeInTheDocument());
    expect(screen.queryByText("● live")).not.toBeInTheDocument();
  });
});
