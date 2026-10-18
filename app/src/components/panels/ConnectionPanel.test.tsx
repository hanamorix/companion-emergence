// Component tests for ConnectionPanel — addresses 2026-05-08 audit P4-2
// (frontend test coverage beyond smoke). Verifies that the status
// banner surfaces real degraded modes and state-poll errors so the
// audit's "user can't tell something's broken" scenario can't regress.
//
// Also covers the auto-update UI (Phase 4): idle, checking, up-to-date,
// update available, error, download, and ready states.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// ── mocks ──────────────────────────────────────────────────────────

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
  supportsNellCliShortcut: (platform: string) => platform === "macos" || platform === "linux" || platform === "windows",
  detectInstallShape: vi.fn(async () => "native"),
}));

const { check } = vi.hoisted(() => ({
  check: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-updater", () => ({
  check,
  // Update class for the "available" state
  Update: class {},
}));

vi.mock("@tauri-apps/api/app", () => ({
  getVersion: vi.fn(async () => "0.0.42"),
}));

vi.mock("./BrainUpdateRow", () => ({
  BrainUpdateRow: ({ checkToken }: { checkToken: number }) => (
    <div data-testid="brain-row">token {checkToken}</div>
  ),
}));

import { ConnectionPanel } from "./ConnectionPanel";
import type { PersonaState } from "../../bridge";
import { getClientPlatform, detectInstallShape } from "../../platform";
import type { Update } from "@tauri-apps/plugin-updater";
import { invoke } from "@tauri-apps/api/core";
import { getVersion } from "@tauri-apps/api/app";

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

function fakeUpdate(version = "0.0.12"): Update {
  return {
    version,
    body: "release notes",
    date: "2026-05-14T00:00:00Z",
    downloadAndInstall: vi.fn(),
  } as unknown as Update;
}

// ── StatusBanner tests ─────────────────────────────────────────────

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

  it("shows the Task Scheduler supervisor button (not launchd) and the nell shortcut button on Windows", () => {
    vi.mocked(getClientPlatform).mockReturnValue("windows");
    render(<ConnectionPanel state={baseState()} persona="test" />);

    // Supervisor is now supported on Windows via Task Scheduler — launchd button stays macOS-only,
    // but the Windows supervisor button IS present.
    expect(screen.queryByRole("button", { name: /install launchd supervisor/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /install Task Scheduler supervisor/i })).toBeInTheDocument();
    // The Terminal shortcut now exists on Windows too (#408).
    expect(screen.getByRole("button", { name: /install nell to your PATH/i })).toBeInTheDocument();
    expect(screen.queryByText(/macOS-only right now/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No action needed for normal app use/i)).not.toBeInTheDocument();
  });

  it("shows calm unsupported notes instead of macOS install buttons on Linux (supervisor now supported on Linux — this test updated in Phase 3.2)", () => {
    vi.mocked(getClientPlatform).mockReturnValue("linux");
    render(<ConnectionPanel state={baseState()} persona="test" />);

    // Linux now has its own systemd install button — launchd button should not appear
    expect(screen.queryByRole("button", { name: /install launchd supervisor/i })).not.toBeInTheDocument();
    // But the systemd button IS present
    expect(screen.getByRole("button", { name: /install systemd supervisor/i })).toBeInTheDocument();
    // And so is the nell shortcut button, enabled (#408)
    const cli = screen.getByRole("button", { name: /install nell to ~\/\.local\/bin/i });
    expect(cli).toBeEnabled();
    expect(screen.queryByText(/No action needed for normal app use/i)).not.toBeInTheDocument();
  });

  it("renders the privacy row marked accent (local-only)", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByText(/local-only/i)).toBeInTheDocument();
  });

  // v0.0.14: bridge restart button integration. The button must only
  // appear inside StatusBanner when kind === "error" (bridge_down /
  // offline / stateError) — warning-kind banners (provider_down) and
  // healthy state should not surface it.
  it("renders RestartBridgeButton inside the banner when kind === error", () => {
    render(
      <ConnectionPanel state={baseState({ mode: "bridge_down" })} persona="test" />,
    );
    const banner = screen.getByRole("alert");
    expect(banner).toHaveTextContent(/end conversation and restart/i);
  });

  it("does not render RestartBridgeButton when mode is healthy", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(
      screen.queryByText(/end conversation and restart/i),
    ).not.toBeInTheDocument();
  });

  it("does not render RestartBridgeButton on warning-kind banners (provider_down)", () => {
    render(
      <ConnectionPanel
        state={baseState({ mode: "provider_down" })}
        persona="test"
      />,
    );
    const banner = screen.getByRole("alert");
    expect(banner).toBeInTheDocument();
    expect(
      screen.queryByText(/end conversation and restart/i),
    ).not.toBeInTheDocument();
  });

  it('does not say "macOS-only" for unsupported platform', () => {
    vi.mocked(getClientPlatform).mockReturnValue("other" as never);
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.queryByText(/macOS-only/i)).not.toBeInTheDocument();
  });
});

// ── Footer aphorism (glass redesign, Phase 3) ───────────────────────

describe("ConnectionPanel — footer aphorism", () => {
  afterEach(() => cleanup());

  it("renders the persona-interpolated aphorism footer", () => {
    render(<ConnectionPanel state={baseState()} persona="nell" />);
    expect(
      screen.getByText("You configure the room. Nell owns the weather."),
    ).toBeInTheDocument();
  });
});

// ── Update UI tests (Phase 4) ──────────────────────────────────────

describe("ConnectionPanel — UpdateSection (Phase 4)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
    vi.mocked(check).mockReset();
  });
  afterEach(() => {
    cleanup();
  });

  it("shows Check for updates button when idle", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByRole("button", { name: /check for updates/i })).toBeInTheDocument();
  });

  it("shows checking state after clicking check", async () => {
    // check() hangs → stays in checking state
    vi.mocked(check).mockImplementation(() => new Promise(() => {}));
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText(/checking for updates/i)).toBeInTheDocument();
    });
  });

  it("shows up-to-date when check returns null", async () => {
    vi.mocked(check).mockResolvedValue(null);
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText(/up to date/i)).toBeInTheDocument();
    });
  });

  it("shows update available when check returns an update", async () => {
    vi.mocked(check).mockResolvedValue(fakeUpdate("0.0.12"));
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText(/0.0.12 available/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /download/i })).toBeInTheDocument();
  });

  it("shows error state on check failure", async () => {
    vi.mocked(check).mockRejectedValue(new Error("network error"));
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText(/could not check for updates/i)).toBeInTheDocument();
    });
    // Retry button
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("shows downloading state after clicking Download & Install", async () => {
    const update = fakeUpdate("0.0.12");
    vi.mocked(check).mockResolvedValue(update);
    // downloadAndInstall hangs → stays in downloading
    vi.mocked(update.downloadAndInstall).mockImplementation(() => new Promise(() => {}));

    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText(/0.0.12 available/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /download/i }));
    await waitFor(() => {
      expect(screen.getByText(/downloading update/i)).toBeInTheDocument();
    });
  });
});

// ── Install-shape gating (Phase 2) ────────────────────────────────

describe("ConnectionPanel — install shape gating (Phase 2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("linux");
    vi.mocked(check).mockReset();
  });
  afterEach(() => {
    cleanup();
  });

  it("shows release-page link instead of Download button when shape is deb", async () => {
    vi.mocked(detectInstallShape).mockResolvedValue("deb");
    vi.mocked(check).mockResolvedValue(fakeUpdate("0.0.15"));

    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));

    await waitFor(() => {
      expect(screen.getByText(/0.0.15 available/i)).toBeInTheDocument();
    });

    // shape resolves after check — wait for the link to appear
    await waitFor(() => {
      expect(screen.getByRole("link", { name: /releases page/i })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /download/i })).not.toBeInTheDocument();
  });

  it("shows Download & Install button when shape is appimage", async () => {
    vi.mocked(detectInstallShape).mockResolvedValue("appimage");
    vi.mocked(check).mockResolvedValue(fakeUpdate("0.0.15"));

    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));

    await waitFor(() => {
      expect(screen.getByText(/0.0.15 available/i)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /download/i })).toBeInTheDocument();
    });
    expect(screen.queryByRole("link", { name: /releases page/i })).not.toBeInTheDocument();
  });
});

// ── Model section (Phase 8.3) ─────────────────────────────────────

vi.mock("./ModelPicker", () => ({
  ModelPicker: ({
    onClose,
    allowReapply,
  }: {
    allowReapply?: boolean;
    onClose: (
      m?: string,
      info?: {
        checked?: false;
        fallbacks?: { family: string; wanted: string; using: string }[];
        tiersUnchecked?: string[];
      },
    ) => void;
  }) => (
    <div data-testid="model-picker" data-allow-reapply={String(allowReapply === true)}>
      <button onClick={() => onClose("opus")}>Apply opus</button>
      <button onClick={() => onClose("claude-sonnet-5", { checked: false })}>Apply unchecked</button>
      <button
        onClick={() =>
          onClose("claude-sonnet-5", { fallbacks: [{ family: "haiku", wanted: "claude-haiku-5", using: "haiku" }] })
        }
      >
        Apply with fallback
      </button>
      <button onClick={() => onClose("claude-sonnet-5", { tiersUnchecked: ["claude-haiku-5"] })}>
        Apply with unchecked tiers
      </button>
      <button onClick={() => onClose()}>Cancel</button>
    </div>
  ),
}));

describe("ConnectionPanel — Model section (Phase 8.3)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
  });
  afterEach(() => {
    cleanup();
  });

  it("shows the current model from state.connection.model", () => {
    render(<ConnectionPanel state={baseState({ connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null } })} persona="test" />);
    // "Currently sonnet." text appears in the Model section (multiple elements contain "sonnet" — use getAllByText)
    expect(screen.getAllByText(/sonnet/i).length).toBeGreaterThan(0);
    // The "change model" button is present
    expect(screen.getByRole("button", { name: /change model/i })).toBeInTheDocument();
  });

  it("shows 'change model' button that reveals ModelPicker", () => {
    render(<ConnectionPanel state={baseState()} persona="test" />);
    const changeBtn = screen.getByRole("button", { name: /change model/i });
    expect(changeBtn).toBeInTheDocument();

    fireEvent.click(changeBtn);
    expect(screen.getByTestId("model-picker")).toBeInTheDocument();
  });

  it("updates local model display and hides picker after apply", async () => {
    render(<ConnectionPanel state={baseState({ connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null } })} persona="test" />);

    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply opus/i }));

    await waitFor(() => {
      expect(screen.queryByTestId("model-picker")).not.toBeInTheDocument();
    });
    // Local model override applied
    expect(screen.getByText(/opus/)).toBeInTheDocument();
  });
});

// ── Rejected / unchecked model (#440) ─────────────────────────────

describe("ConnectionPanel — rejected and unchecked models (#440)", () => {
  const rec = (model: string) => ({
    model,
    since: "2026-10-17T12:00:00+00:00",
    last_at: "2026-10-17T12:05:00+00:00",
    count: 3,
  });
  const sonnetState = (extra: Partial<PersonaState> = {}) =>
    baseState({
      connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null },
      ...extra,
    });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
  });
  afterEach(() => {
    cleanup();
  });

  it("names the chosen model Claude rejected and says what to do", () => {
    render(<ConnectionPanel state={sonnetState({ model_rejections: [rec("sonnet")] })} persona="test" />);
    const line = screen.getByText(/Claude rejected the model "sonnet"/);
    expect(line).toHaveTextContent(/may not exist/i);
    expect(line).toHaveTextContent(/use change to pick another model/i);
    expect(line.textContent).not.toContain("—");
  });

  it("shows nothing when there are no rejections (or the field is missing)", () => {
    const { rerender } = render(<ConnectionPanel state={sonnetState({ model_rejections: [] })} persona="test" />);
    expect(screen.queryByText(/Claude rejected/)).not.toBeInTheDocument();
    rerender(<ConnectionPanel state={sonnetState()} persona="test" />);
    expect(screen.queryByText(/Claude rejected/)).not.toBeInTheDocument();
  });

  it("words a rejected background-only model differently (changing the chat model would not help)", () => {
    render(<ConnectionPanel state={sonnetState({ model_rejections: [rec("claude-tier-9")] })} persona="test" />);
    const line = screen.getByText(/Claude rejected the model "claude-tier-9"/);
    expect(line).toHaveTextContent(/background jobs/i);
    expect(line).not.toHaveTextContent(/pick another model/i);
  });

  it("drops the line as soon as the user changes the model", async () => {
    render(<ConnectionPanel state={sonnetState({ model_rejections: [rec("sonnet")] })} persona="test" />);
    expect(screen.getByText(/Claude rejected the model "sonnet"/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply opus/i }));
    await waitFor(() => expect(screen.queryByTestId("model-picker")).not.toBeInTheDocument());
    expect(screen.queryByText(/Claude rejected/)).not.toBeInTheDocument();
  });

  it("keeps the unconfirmed notice while the bridge lists the model, and clears it when a clean reply arrives", () => {
    const unconfirmed = (models: string[]) =>
      baseState({
        connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
        model_unconfirmed: models,
      });
    const { rerender } = render(<ConnectionPanel state={unconfirmed(["claude-sonnet-5"])} persona="test" />);
    expect(screen.getByText(/Couldn't check that model right now/)).toBeInTheDocument();

    // a clean reply for the model arrives: the bridge's next state no longer lists it
    rerender(<ConnectionPanel state={unconfirmed([])} persona="test" />);
    expect(screen.queryByText(/Couldn't check that model/)).not.toBeInTheDocument();
  });

  it("does not show the unconfirmed notice for a model that is not the displayed one", () => {
    render(
      <ConnectionPanel
        state={sonnetState({ model_unconfirmed: ["claude-sonnet-5"] })}
        persona="test"
      />,
    );
    expect(screen.queryByText(/Couldn't check that model/)).not.toBeInTheDocument();
  });

  const HAIKU = { family: "haiku", wanted: "claude-haiku-5", using: "haiku" };
  const PINNED = { family: "haiku", wanted: "claude-haiku-5", using: "claude-haiku-4-5-20251001" };

  it("shows a plain line for each background tier that fell back (#441)", () => {
    render(
      <ConnectionPanel
        state={baseState({
          connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
          tier_fallbacks: [HAIKU, PINNED],
        })}
        persona="test"
      />,
    );
    const first = screen.getByText(/using "haiku"/);
    expect(first).toHaveTextContent(/Background jobs that use the haiku family are using "haiku"/);
    expect(first).toHaveTextContent(/"claude-haiku-5" could not be used \(it may not exist, or your account may not have access to it\)/);
    expect(first.textContent).not.toContain("—");
    expect(screen.getByText(/using "claude-haiku-4-5-20251001"/)).toBeInTheDocument();
  });

  it("shows the fallbacks the last Apply reported at once, and the bridge's list takes over", async () => {
    const { rerender } = render(
      <ConnectionPanel state={baseState({ connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null } })} persona="test" />,
    );
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply with fallback/i }));
    expect(await screen.findByText(/using "haiku"/)).toBeInTheDocument();
    // the next poll arrives without any fallback: the optimistic line goes away
    rerender(
      <ConnectionPanel state={baseState({ connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null }, tier_fallbacks: [] })} persona="test" />,
    );
    expect(screen.queryByText(/using "haiku"/)).not.toBeInTheDocument();
  });

  it("prefers the fresh Apply result over a stale poll", async () => {
    // the poll still lists a fallback from the previous chat model; Apply reports none: the line goes at once
    render(
      <ConnectionPanel
        state={baseState({
          connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
          tier_fallbacks: [HAIKU],
        })}
        persona="test"
      />,
    );
    expect(screen.getByText(/using "haiku"/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply opus/i }));
    await waitFor(() => expect(screen.queryByText(/using "haiku"/)).not.toBeInTheDocument());
  });

  it("a late pre-Apply poll does not bring back the old lines; the fresh poll takes over", async () => {
    const staleState = baseState({
      connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null },
      tier_fallbacks: [PINNED],
    });
    const { rerender } = render(<ConnectionPanel state={staleState} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply with fallback/i })); // applies claude-sonnet-5
    expect(await screen.findByText(/using "haiku"/)).toBeInTheDocument();
    // a poll that left before the Apply lands afterwards: still the old chat model, old list
    rerender(<ConnectionPanel state={{ ...staleState }} persona="test" />);
    expect(screen.getByText(/using "haiku",/)).toBeInTheDocument();
    expect(screen.queryByText(/using "claude-haiku-4-5-20251001"/)).not.toBeInTheDocument();
    // the first poll that reports the new chat model carries the bridge's own list
    rerender(
      <ConnectionPanel
        state={baseState({
          connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
          tier_fallbacks: [PINNED],
        })}
        persona="test"
      />,
    );
    expect(screen.getByText(/using "claude-haiku-4-5-20251001"/)).toBeInTheDocument();
    expect(screen.queryByText(/using "haiku",/)).not.toBeInTheDocument();
  });

  it("releases the pre-Apply guard at the first poll that reports a model other than the old one", async () => {
    const before = baseState({ connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null }, tier_fallbacks: [] });
    const { rerender } = render(<ConnectionPanel state={before} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply with fallback/i })); // applies claude-sonnet-5
    expect(await screen.findByText(/using "haiku",/)).toBeInTheDocument();
    // the model was changed somewhere else meanwhile: the poll reports neither the old nor the applied model
    rerender(
      <ConnectionPanel
        state={baseState({
          connection: { provider: "claude-cli", model: "claude-opus-4-8", last_heartbeat_at: null },
          tier_fallbacks: [PINNED],
        })}
        persona="test"
      />,
    );
    expect(screen.queryByText(/using "haiku",/)).not.toBeInTheDocument(); // the stale optimistic lines are gone
    expect(screen.getByText(/using "claude-haiku-4-5-20251001"/)).toBeInTheDocument(); // the bridge's list shows
  });

  it("releases the guard after a bounded time even if no poll ever reports a new model", async () => {
    vi.useFakeTimers();
    try {
      const stale = baseState({
        connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null },
        tier_fallbacks: [PINNED],
      });
      const { rerender } = render(<ConnectionPanel state={stale} persona="test" />);
      fireEvent.click(screen.getByRole("button", { name: /change model/i }));
      fireEvent.click(screen.getByRole("button", { name: /apply with fallback/i }));
      expect(screen.getByText(/using "haiku",/)).toBeInTheDocument();
      rerender(<ConnectionPanel state={{ ...stale }} persona="test" />); // still the old model: guard holds
      expect(screen.getByText(/using "haiku",/)).toBeInTheDocument();
      act(() => {
        vi.advanceTimersByTime(15_000);
      });
      expect(screen.queryByText(/using "haiku",/)).not.toBeInTheDocument();
      expect(screen.getByText(/using "claude-haiku-4-5-20251001"/)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("holds a same-model re-Apply against a stale poll until the bounded time ends", () => {
    vi.useFakeTimers();
    try {
      const same = baseState({
        connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
        tier_fallbacks: [PINNED],
      });
      const { rerender } = render(<ConnectionPanel state={same} persona="test" />);
      fireEvent.click(screen.getByRole("button", { name: /change model/i }));
      fireEvent.click(screen.getByRole("button", { name: /apply with fallback/i })); // re-applies claude-sonnet-5
      expect(screen.getByText(/using "haiku",/)).toBeInTheDocument();
      rerender(<ConnectionPanel state={{ ...same }} persona="test" />); // a poll that left before the Apply
      expect(screen.getByText(/using "haiku",/)).toBeInTheDocument(); // still the Apply's answer
      expect(screen.queryByText(/using "claude-haiku-4-5-20251001"/)).not.toBeInTheDocument();
      act(() => {
        vi.advanceTimersByTime(15_000);
      });
      expect(screen.queryByText(/using "haiku",/)).not.toBeInTheDocument();
      expect(screen.getByText(/using "claude-haiku-4-5-20251001"/)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("says plainly which background models were not checked yet, at once and from the poll, and clears it", async () => {
    const withModel = (extra: Partial<PersonaState>) =>
      baseState({ connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null }, ...extra });
    const { rerender } = render(<ConnectionPanel state={sonnetState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply with unchecked tiers/i }));
    const note = await screen.findByText(/were not checked yet/);
    expect(note).toHaveTextContent(/claude-haiku-5/);
    expect(note).toHaveTextContent(/checked when first used, and any problem will show here/);
    expect(note.textContent).not.toContain("—");
    // the bridge's poll carries it next
    rerender(<ConnectionPanel state={withModel({ tier_unconfirmed: ["claude-haiku-5"] })} persona="test" />);
    expect(screen.getByText(/were not checked yet/)).toBeInTheDocument();
    // a clean reply for that id arrives: the next poll no longer lists it
    rerender(<ConnectionPanel state={withModel({ tier_unconfirmed: [] })} persona="test" />);
    expect(screen.queryByText(/were not checked yet/)).not.toBeInTheDocument();
  });

  it("does not repeat a rejection line that a fallback line already explains", () => {
    render(
      <ConnectionPanel
        state={baseState({
          connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
          tier_fallbacks: [HAIKU],
          model_rejections: [
            { model: "claude-haiku-5", since: "t", last_at: "t", count: 2 },
          ],
        })}
        persona="test"
      />,
    );
    expect(screen.getByText(/using "haiku"/)).toBeInTheDocument();
    expect(screen.queryByText(/Claude rejected the model "claude-haiku-5"/)).not.toBeInTheDocument();
  });

  it("never hides a rejection of the chat model, even when a fallback names the same id", () => {
    render(
      <ConnectionPanel
        state={baseState({
          connection: { provider: "claude-cli", model: "claude-sonnet-5", last_heartbeat_at: null },
          tier_fallbacks: [{ family: "sonnet", wanted: "claude-sonnet-5", using: "sonnet" }],
          model_rejections: [{ model: "claude-sonnet-5", since: "t", last_at: "t", count: 2 }],
        })}
        persona="test"
      />,
    );
    expect(screen.getByText(/Claude rejected the model "claude-sonnet-5"/)).toBeInTheDocument();
  });

  it("renders nothing for fallbacks when there are none", () => {
    render(<ConnectionPanel state={sonnetState({ tier_fallbacks: [] })} persona="test" />);
    expect(screen.queryByText(/Background jobs that use/)).not.toBeInTheDocument();
  });

  it("tells the user plainly when a custom model was saved without being checked, and clears it on the next change", async () => {
    render(<ConnectionPanel state={sonnetState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply unchecked/i }));
    const notice = await screen.findByText(/Couldn't check that model right now/);
    expect(notice).toHaveTextContent(/saved anyway, and any problem will show here/);
    expect(notice.textContent).not.toContain("—");

    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply opus/i }));
    await waitFor(() => expect(screen.queryByText(/Couldn't check that model/)).not.toBeInTheDocument());
  });
});

// ── Supervisor install (Phase 3.2) ────────────────────────────────

describe("ConnectionPanel — supervisor install (Phase 3.2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(check).mockReset();
  });
  afterEach(() => {
    cleanup();
  });

  it("Linux: renders systemd copy in the supervisor section", () => {
    vi.mocked(getClientPlatform).mockReturnValue("linux");
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByText(/systemd --user/i)).toBeInTheDocument();
  });

  it("Linux: install button is rendered and enabled", () => {
    vi.mocked(getClientPlatform).mockReturnValue("linux");
    render(<ConnectionPanel state={baseState()} persona="test" />);
    const btn = screen.getByRole("button", { name: /install systemd supervisor/i });
    expect(btn).toBeInTheDocument();
    expect(btn).toBeEnabled();
  });

  it("macOS: still renders launchd copy and button", () => {
    vi.mocked(getClientPlatform).mockReturnValue("macos");
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByText(/launchd LaunchAgent/i)).toBeInTheDocument();
    const btn = screen.getByRole("button", { name: /install launchd supervisor/i });
    expect(btn).toBeInTheDocument();
    expect(btn).toBeEnabled();
  });

  it("shows the Task Scheduler supervisor button on Windows", () => {
    vi.mocked(getClientPlatform).mockReturnValue("windows");
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByText(/install Task Scheduler supervisor/i)).toBeInTheDocument();
  });
});

// ── Clean connection / brain login (Task 4) ─────────────────────────

describe("ConnectionPanel — Clean connection section", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
  });
  afterEach(() => {
    cleanup();
  });

  it("shows an Authorize control when the brain is not yet authorized", async () => {
    vi.mocked(invoke).mockImplementation(async (cmd: string) => {
      if (cmd === "brain_login_status") return { authorized: false };
      return { success: true, stdout: "", stderr: "", exit_code: 0 };
    });
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(
      await screen.findByRole("button", { name: /authorize/i }),
    ).toBeInTheDocument();
  });
});

// ── Brain update row + app version (Task 7) ─────────────────────────

describe("ConnectionPanel — BrainUpdateRow wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
    vi.mocked(check).mockReset();
  });
  afterEach(() => {
    cleanup();
  });

  it("renders BrainUpdateRow at token 0 and bumps it once per Check for updates click", () => {
    vi.mocked(check).mockImplementation(() => new Promise(() => {}));
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.getByTestId("brain-row")).toHaveTextContent("token 0");

    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    expect(screen.getByTestId("brain-row")).toHaveTextContent("token 1");
  });
});

// ── App version alongside app-update text (Task 7) ──────────────────

describe("ConnectionPanel — app version in the update-available text", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
    vi.mocked(check).mockReset();
  });
  afterEach(() => {
    cleanup();
  });

  it("shows the app's current version alongside an available update", async () => {
    vi.mocked(check).mockResolvedValue(fakeUpdate("0.0.43"));
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText("v0.0.43 available. Current: v0.0.42.")).toBeInTheDocument();
    });
  });

  it("omits the current-version clause when getVersion fails", async () => {
    vi.mocked(getVersion).mockRejectedValueOnce(new Error("no version"));
    vi.mocked(check).mockResolvedValue(fakeUpdate("0.0.43"));
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /check for updates/i }));
    await waitFor(() => {
      expect(screen.getByText("v0.0.43 available.")).toBeInTheDocument();
    });
  });
});





// ── nell shortcut button on every supported platform (#408, C14 / C29b) ──────────

const NO_FUSE_TEXT =
  "This AppImage is running without FUSE (in extract-and-run mode), so the nell shortcut was not installed. A shortcut would re-extract the whole app on every nell command and delete the running app's files when the command ends. To get the shortcut, install FUSE for your distribution (on Debian and Ubuntu the package is libfuse2, called libfuse2t64 on Ubuntu 24.04), start the AppImage normally, and try again. Or install the .deb package instead, which does not need FUSE.";

describe("ConnectionPanel: nell shortcut button", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  afterEach(() => {
    cleanup();
  });

  it.each([
    ["linux", /install nell to ~\/\.local\/bin/i],
    ["windows", /install nell to your PATH/i],
    ["macos", /install nell to ~\/\.local\/bin/i],
  ] as Array<[string, RegExp]>)("%s: button is shown, enabled and clicking invokes the command once", async (platform, name) => {
    vi.mocked(getClientPlatform).mockReturnValue(platform as never);
    vi.mocked(invoke).mockImplementation((async (cmd: string) =>
      cmd === "install_nell_cli_symlink"
        ? { success: true, stdout: "nell shortcut installed", stderr: "", exit_code: 0 }
        : { success: true, stdout: "service installed", stderr: "", exit_code: 0 }) as never);
    render(<ConnectionPanel state={baseState()} persona="test" />);
    const btn = screen.getByRole("button", { name });
    expect(btn).toBeEnabled();
    fireEvent.click(btn);
    await waitFor(() => expect(screen.getByText(/nell shortcut installed/i)).toBeInTheDocument());
    const calls = vi.mocked(invoke).mock.calls.filter((c) => c[0] === "install_nell_cli_symlink");
    expect(calls).toHaveLength(1);
  });

  it("other platforms keep the calm note and have no button", () => {
    vi.mocked(getClientPlatform).mockReturnValue("other" as never);
    render(<ConnectionPanel state={baseState()} persona="test" />);
    expect(screen.queryByRole("button", { name: /install nell to/i })).not.toBeInTheDocument();
    expect(screen.getByText(/No action needed for normal app use/i)).toBeInTheDocument();
  });

  it("linux: the no-FUSE explanation is shown in full as the error detail", async () => {
    vi.mocked(getClientPlatform).mockReturnValue("linux" as never);
    vi.mocked(invoke).mockImplementation((async (cmd: string) =>
      cmd === "install_nell_cli_symlink"
        ? { success: false, stdout: "", stderr: NO_FUSE_TEXT, exit_code: 78 }
        : { success: true, stdout: "", stderr: "", exit_code: 0 }) as never);
    render(<ConnectionPanel state={baseState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /install nell to ~\/\.local\/bin/i }));
    await waitFor(() => expect(screen.getByText(NO_FUSE_TEXT)).toBeInTheDocument());
  });
});

// ── Unsupported model setting (#446) ──────────────────────────────

describe("ConnectionPanel — unsupported model setting (#446)", () => {
  const sonnetState = (extra: Partial<PersonaState> = {}) =>
    baseState({
      connection: { provider: "claude-cli", model: "sonnet", last_heartbeat_at: null },
      ...extra,
    });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getClientPlatform).mockReturnValue("macos");
  });
  afterEach(() => {
    cleanup();
  });

  it("shows a plain line for an unsupported model setting, naming the value and Sonnet", () => {
    render(<ConnectionPanel state={sonnetState({ model_unsupported: "fable" })} persona="test" />);
    const line = screen.getByText(/The model "fable" in your settings is not supported/);
    expect(line).toHaveTextContent(/Sonnet is being used instead/);
    expect(line).toHaveAttribute("role", "status");
    expect(line.textContent).not.toContain("—");
  });

  it("shows no unsupported-model line when the field is null or missing", () => {
    const { rerender } = render(<ConnectionPanel state={sonnetState({ model_unsupported: null })} persona="test" />);
    expect(screen.queryByText(/is not supported/)).not.toBeInTheDocument();
    rerender(<ConnectionPanel state={sonnetState()} persona="test" />);
    expect(screen.queryByText(/is not supported/)).not.toBeInTheDocument();
  });

  it("tells the picker to allow re-applying the current model only while the unsupported line is shown", () => {
    // ModelPicker is mocked in this file; its own allowReapply behaviour is tested in ModelPicker.test.tsx.
    const { unmount } = render(<ConnectionPanel state={sonnetState({ model_unsupported: "fable" })} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    expect(screen.getByTestId("model-picker")).toHaveAttribute("data-allow-reapply", "true");
    unmount();
    render(<ConnectionPanel state={sonnetState()} persona="test" />);
    fireEvent.click(screen.getByRole("button", { name: /change model/i }));
    expect(screen.getByTestId("model-picker")).toHaveAttribute("data-allow-reapply", "false");
  });
});
