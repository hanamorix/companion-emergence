// Tests for BrainUpdateRow (#286 slice 4, Task 7) — the brain-update row
// under ConnectionPanel's "Check for updates". Mocks useBrainUpdate so
// every state the hook can report is exercised without touching the
// network/bridge.

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

const { useBrainUpdate, check, apply, useReleaseBrain } = vi.hoisted(() => ({
  useBrainUpdate: vi.fn(),
  check: vi.fn(async () => {}),
  apply: vi.fn(async () => {}),
  useReleaseBrain: vi.fn(async () => {}),
}));

vi.mock("../../hooks/useBrainUpdate", () => ({ useBrainUpdate }));

import { BrainUpdateRow } from "./BrainUpdateRow";
import type { BrainUpdateState } from "../../hooks/useBrainUpdate";
import type { BridgeOverlay } from "../../bridge";

function setHook(state: BrainUpdateState, overlay: BridgeOverlay | null = null) {
  useBrainUpdate.mockReturnValue({ state, overlay, check, apply, useReleaseBrain });
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("BrainUpdateRow — check() on checkToken", () => {
  it("does not check on mount, checks once per token increase", () => {
    setHook({ kind: "idle" });
    const { rerender } = render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(check).not.toHaveBeenCalled();

    rerender(<BrainUpdateRow persona="nell" mode="live" checkToken={1} />);
    expect(check).toHaveBeenCalledTimes(1);

    rerender(<BrainUpdateRow persona="nell" mode="live" checkToken={2} />);
    expect(check).toHaveBeenCalledTimes(2);
  });
});

describe("BrainUpdateRow — running overlay", () => {
  it("shows the running overlay and a working 'Use the release brain' button", () => {
    setHook({ kind: "idle" }, { commit: "abc1234def5678", brain_version: "0.0.44" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Brain: main @abc1234 (0.0.44)")).toBeInTheDocument();
    const btn = screen.getByRole("button", { name: /use the release brain/i });
    fireEvent.click(btn);
    expect(useReleaseBrain).toHaveBeenCalledTimes(1);
  });

  it("shows an overlay whose build is unknown, with the release-brain button", () => {
    setHook({ kind: "idle" }, { commit: null, brain_version: null });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Brain: main (unknown build)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /use the release brain/i })).toBeEnabled();
  });

  it("disables 'Use the release brain' while checking", () => {
    setHook({ kind: "checking" }, { commit: "abc1234def5678", brain_version: "0.0.44" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByRole("button", { name: /use the release brain/i })).toBeDisabled();
  });

  it("renders nothing visible when there is no overlay and state is idle", () => {
    setHook({ kind: "idle" }, null);
    const { container } = render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("BrainUpdateRow — available update", () => {
  it("shows the available-update text and a working apply button", () => {
    setHook({ kind: "available", commit: "c".repeat(40), brainVersion: "0.0.44" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(
      screen.getByText("Brain update available: main @ccccccc (0.0.44)"),
    ).toBeInTheDocument();
    const btn = screen.getByRole("button", { name: /end conversation and update the brain/i });
    fireEvent.click(btn);
    expect(apply).toHaveBeenCalledTimes(1);
  });
});

describe("BrainUpdateRow — busy states", () => {
  it("checking", () => {
    setHook({ kind: "checking" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Checking for a brain update…")).toBeInTheDocument();
  });

  it("applying", () => {
    setHook({ kind: "applying", commit: "abc1234" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Installing the brain update…")).toBeInTheDocument();
  });

  it("restarting", () => {
    setHook({ kind: "restarting" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Restarting the brain…")).toBeInTheDocument();
  });

  it("reverting", () => {
    setHook({ kind: "reverting" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Switching to the release brain…")).toBeInTheDocument();
  });
});

describe("BrainUpdateRow — none", () => {
  it("app_too_old", () => {
    setHook({ kind: "none", reason: "app_too_old" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(
      screen.getByText("A brain update needs a newer app — update Companion Emergence first."),
    ).toBeInTheDocument();
  });

  it("unsupported_manifest asks for a newer app", () => {
    setHook({ kind: "none", reason: "unsupported_manifest" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(
      screen.getByText("A brain update needs a newer app — update Companion Emergence first."),
    ).toBeInTheDocument();
  });

  it.each(["unsupported_install", "unparseable_version"])("%s says no updates for this install", (reason) => {
    setHook({ kind: "none", reason });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("No brain updates for this install.")).toBeInTheDocument();
  });

  it("already_active", () => {
    setHook({ kind: "none", reason: "already_active" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("The brain is up to date ✓")).toBeInTheDocument();
  });

  it("any other reason", () => {
    setHook({ kind: "none", reason: "some_other_reason" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("The brain is up to date ✓")).toBeInTheDocument();
  });

  it("dev_build renders nothing", () => {
    setHook({ kind: "none", reason: "dev_build" });
    const { container } = render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("BrainUpdateRow — live status", () => {
  it("announces the status line politely", () => {
    setHook({ kind: "checking" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Checking for a brain update…").closest('[aria-live="polite"]')).not.toBeNull();
  });
});

describe("BrainUpdateRow — rolled_back", () => {
  it("shows the rollback notice with role status", () => {
    setHook({ kind: "rolled_back" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    const el = screen.getByRole("status");
    expect(el).toHaveTextContent("The new brain didn't start, so it was rolled back.");
  });
});

describe("BrainUpdateRow — error", () => {
  it("shows the hook's detail sentence as-is", () => {
    setHook({ kind: "error", detail: "Couldn't update the brain: x" });
    render(<BrainUpdateRow persona="nell" mode="live" checkToken={0} />);
    expect(screen.getByText("Couldn't update the brain: x")).toBeInTheDocument();
  });
});
