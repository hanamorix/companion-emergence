// Boot-phase routing tests for App.tsx.
// Covers the four branches introduced in Bundle 9:
//   1. selected_persona set → skips listPersonas, starts bridge (no picker)
//   2. no selection, 0 personas → wizard
//   3. no selection, 1 persona → auto-selects, writeAppConfig called, no picker
//   4. no selection, ≥2 personas → picker shown

import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
const {
  readAppConfig,
  writeAppConfig,
  listPersonas,
  ensureBridgeRunning,
  setAlwaysOnTop,
  brainLoginStatus,
  brainOverlayStatus,
  revertBrain,
  rollbackBrain,
  confirmBrainUpdate,
  nellbrainHomePath,
} = vi.hoisted(() => ({
  readAppConfig: vi.fn(),
  writeAppConfig: vi.fn(async () => undefined),
  listPersonas: vi.fn(),
  ensureBridgeRunning: vi.fn(async () => undefined),
  setAlwaysOnTop: vi.fn(async () => undefined),
  // Default authorized:true so these boot-routing tests never see the
  // brain-login banner — that's covered separately in App.brainLogin.test.tsx.
  brainLoginStatus: vi.fn(async () => ({ authorized: true })),
  brainOverlayStatus: vi.fn(async () => ({ active_commit: null as string | null, confirmed: true, undo: "revert" })),
  revertBrain: vi.fn(async () => undefined),
  rollbackBrain: vi.fn(async (_reason: string) => undefined),
  confirmBrainUpdate: vi.fn(async (_commit: string) => undefined),
  nellbrainHomePath: vi.fn(async (): Promise<string | null> => null),
}));

vi.mock("./appConfig", () => ({
  readAppConfig,
  writeAppConfig,
  listPersonas,
  ensureBridgeRunning,
  setAlwaysOnTop,
  brainLoginStatus,
  brainOverlayStatus,
  revertBrain,
  rollbackBrain,
  confirmBrainUpdate,
  nellbrainHomePath,
}));

// ── bridge ────────────────────────────────────────────────────────────────────
import type { PersonaState } from "./bridge";

const { fetchPersonaState, approvePendingWrite, declinePendingWrite } = vi.hoisted(() => {
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
  };
});

vi.mock("./bridge", () => ({
  fetchPersonaState,
  approvePendingWrite,
  declinePendingWrite,
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
import { NOTICE_ROLLED_BACK } from "./brainRecovery";

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
    brainLoginStatus.mockReset().mockResolvedValue({ authorized: true });
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

  it("version_mismatch_unresolved → shows inline version-mismatch notice in ready state", async () => {
    readAppConfig.mockResolvedValue(baseConfig("nell"));
    ensureBridgeRunning.mockResolvedValue(undefined);
    ensureBridgeCurrent.mockResolvedValue("version_mismatch_unresolved");

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/different version/i)).toBeInTheDocument()
    );
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

describe("App pending-write cards", () => {
  beforeEach(() => {
    readAppConfig.mockReset().mockResolvedValue(baseConfig("nell"));
    writeAppConfig.mockReset().mockResolvedValue(undefined);
    listPersonas.mockReset();
    ensureBridgeRunning.mockReset().mockResolvedValue(undefined);
    ensureBridgeCurrent.mockReset().mockResolvedValue("ok");
    setAlwaysOnTop.mockReset().mockResolvedValue(undefined);
    brainLoginStatus.mockReset().mockResolvedValue({ authorized: true });
    approvePendingWrite.mockClear();
    declinePendingWrite.mockClear();
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
      pending_writes: [
        {
          id: "w_1",
          op: "create",
          path: "/Users/h/note.md",
          preview: "draft body",
          truncated: false,
          proposed_at: "2026-06-14T12:00:00+00:00",
        },
      ],
    });
  });

  afterEach(cleanup);

  it("renders a PendingWriteCard and approve calls the bridge helper", async () => {
    const { fireEvent } = await import("@testing-library/react");
    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/note\.md/)).toBeInTheDocument()
    );
    fireEvent.click(screen.getByText(/approve/i));
    await waitFor(() =>
      expect(approvePendingWrite).toHaveBeenCalledWith("nell", "w_1")
    );
  });
});

describe("App presence column (glass redesign)", () => {
  beforeEach(() => {
    readAppConfig.mockReset().mockResolvedValue(baseConfig("nell"));
    writeAppConfig.mockReset().mockResolvedValue(undefined);
    listPersonas.mockReset();
    ensureBridgeRunning.mockReset().mockResolvedValue(undefined);
    ensureBridgeCurrent.mockReset().mockResolvedValue("ok");
    setAlwaysOnTop.mockReset().mockResolvedValue(undefined);
    brainLoginStatus.mockReset().mockResolvedValue({ authorized: true });
  });

  afterEach(cleanup);

  it("derives a humanized status line from the top-2 emotions", async () => {
    fetchPersonaState.mockReset().mockResolvedValue({
      persona: "nell",
      emotions: { creative_hunger: 8.2, rest_need: 6.1, joy: 1.0 },
      body: null,
      interior: { dream: null, research: null, heartbeat: null, reflex: null },
      soul_highlight: null,
      connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
      mode: "live",
      recovering: false,
      felt_time_recovered: false,
    });

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText(/creative hunger/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/rest need/)).toBeInTheDocument();
  });

  it("shows the background-work label and one job, cleared when none run", async () => {
    // ram-spike-fix INC-11 follow-up (spec §6, S15/S23/S38; owner-set
    // wording + layout). One running job: the label line, then
    // "<name> (<descriptor>)" on its own line below it.
    const runningState = (jobs: { name: string; running_for_seconds: number }[]) => ({
      persona: "nell",
      emotions: {},
      body: null,
      interior: { dream: null, research: null, heartbeat: null, reflex: null },
      soul_highlight: null,
      connection: { provider: "claude-cli", model: null, last_heartbeat_at: null },
      mode: "live" as const,
      recovering: false,
      felt_time_recovered: false,
      background_jobs: jobs,
    });

    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      fetchPersonaState
        .mockReset()
        .mockResolvedValueOnce(runningState([{ name: "compaction", running_for_seconds: 5 }])) // poll 1: running
        .mockResolvedValue(runningState([])); // poll 2+: cleared

      render(<App />);
      await waitFor(() =>
        expect(screen.getByText("Current background tasks:")).toBeInTheDocument(),
      );
      expect(screen.getByText("compaction (conversation compaction)")).toBeInTheDocument();

      await vi.advanceTimersByTimeAsync(5000); // poll 2
      await waitFor(() =>
        expect(screen.queryByText("Current background tasks:")).not.toBeInTheDocument(),
      );
      expect(screen.queryByText(/conversation compaction/)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("renders 3 jobs one per line, longest-running first, in the bridge's given order", async () => {
    // The bridge already orders `background_jobs` longest-running first
    // (brain/bridge/background_jobs.snapshot_with_elapsed) — the app must
    // render that order as given, not re-sort it.
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
      background_jobs: [
        { name: "heartbeat", running_for_seconds: 42.0 }, // longest-running -> first
        { name: "compaction", running_for_seconds: 8.5 },
        { name: "pass2", running_for_seconds: 1.2 }, // most recent -> last
      ],
    });

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText("Current background tasks:")).toBeInTheDocument(),
    );
    const container = screen.getByText("Current background tasks:").parentElement;
    expect(container).not.toBeNull();
    const lines = Array.from(container!.children).map((el) => el.textContent);
    expect(lines).toEqual([
      "Current background tasks:",
      "heartbeat (heartbeat)",
      "compaction (conversation compaction)",
      "pass2 (follow-up memory pass)",
    ]);
  });

  it("renders no background-work line when the field is absent (older bridge)", async () => {
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

    render(<App />);

    await waitFor(() => expect(screen.getByText(/Nell/)).toBeInTheDocument());
    expect(screen.queryByText("Current background tasks:")).not.toBeInTheDocument();
  });
});

describe("App launch recovery (#335)", () => {
  beforeEach(() => {
    readAppConfig.mockReset().mockResolvedValue(baseConfig("nell"));
    ensureBridgeRunning.mockReset();
    brainOverlayStatus.mockReset();
    revertBrain.mockReset().mockResolvedValue(undefined);
    rollbackBrain.mockReset().mockResolvedValue(undefined);
    setAlwaysOnTop.mockReset().mockResolvedValue(undefined);
    brainLoginStatus.mockReset().mockResolvedValue({ authorized: true });
  });

  afterEach(cleanup);

  it("a proven brain update that won't start offers the release brain, which then starts", async () => {
    ensureBridgeRunning.mockRejectedValueOnce(new Error("supervisor_start_timeout")).mockResolvedValue(undefined);
    brainOverlayStatus
      .mockResolvedValueOnce({ active_commit: "a".repeat(40), confirmed: true, undo: "rollback" })
      .mockResolvedValue({ active_commit: null, confirmed: true, undo: "revert" });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Use the release brain" }));

    await waitFor(() => expect(revertBrain).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.queryByText(/Brain startup needs attention/)).not.toBeInTheDocument(),
    );
    expect(rollbackBrain).not.toHaveBeenCalled();
    expect(ensureBridgeRunning).toHaveBeenCalledTimes(2);
  });

  it("switching to the release brain shows progress, not a clickable error screen", async () => {
    ensureBridgeRunning.mockRejectedValueOnce(new Error("supervisor_start_timeout")).mockResolvedValue(undefined);
    brainOverlayStatus
      .mockResolvedValueOnce({ active_commit: "a".repeat(40), confirmed: true, undo: "rollback" })
      .mockResolvedValue({ active_commit: null, confirmed: true, undo: "revert" });
    let release!: () => void;
    revertBrain.mockReturnValueOnce(
      new Promise<undefined>((resolve) => {
        release = () => resolve(undefined);
      }),
    );

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Use the release brain" }));

    expect(await screen.findByText(/Starting brain for nell/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Use the release brain" })).not.toBeInTheDocument();
    release();
    await waitFor(() => expect(ensureBridgeRunning).toHaveBeenCalledTimes(2));
  });

  it("without an active overlay the error screen has no release-brain button", async () => {
    ensureBridgeRunning.mockRejectedValue(new Error("supervisor_start_timeout"));
    brainOverlayStatus.mockResolvedValue({ active_commit: null, confirmed: true, undo: "revert" });

    render(<App />);
    await screen.findByText(/Brain startup needs attention/);
    expect(screen.queryByRole("button", { name: "Use the release brain" })).not.toBeInTheDocument();
  });

  it("an unproven update that won't start rolls back by itself and says so", async () => {
    ensureBridgeRunning.mockRejectedValueOnce(new Error("boom")).mockResolvedValue(undefined);
    brainOverlayStatus
      .mockResolvedValueOnce({ active_commit: "b".repeat(40), confirmed: false, undo: "rollback" })
      .mockResolvedValue({ active_commit: "a".repeat(40), confirmed: true, undo: "revert" });

    render(<App />);
    expect(await screen.findByText(NOTICE_ROLLED_BACK)).toBeInTheDocument();
    expect(rollbackBrain).toHaveBeenCalledWith("unconfirmed overlay failed to start at launch");

    fireEvent.click(screen.getByRole("button", { name: "Dismiss notice" }));
    expect(screen.queryByText(NOTICE_ROLLED_BACK)).not.toBeInTheDocument();
  });
});
