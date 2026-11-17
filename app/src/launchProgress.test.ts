// The launch screen during the ordinary blocking start (HR5 part 5): real launchBrain, real appConfig and real hook state;
// only the Tauri calls are faked. While `ensure_bridge_running` blocks (the bridge's first rebuild), the status the app
// reads shows the engine's record, and the boot screen's StopProgress source (`useRestartBridge`) must show it.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("./bridgeVersionCheck", () => ({ ensureBridgeCurrent: vi.fn(async () => "ok") }));

import { invoke } from "@tauri-apps/api/core";
import { launchBrain } from "./brainRecovery";
import { _resetRestartBridgeForTests, restartInFlight, useRestartBridge } from "./hooks/useRestartBridge";

const mockInvoke = vi.mocked(invoke);
const P = "nell";
const LINE = "Updating your memories: 3 of 9 (33%).";

let startGate: { open: () => void; fail: (e: unknown) => void };
let record: object | null;

function fakeTauri() {
  mockInvoke.mockImplementation(async (cmd) => {
    switch (cmd) {
      case "bridge_status":
        return { alive: "alive", phase: "live", engine_progress: record };
      case "brain_overlay_status":
        return { active_commit: null, confirmed: true, undo: "revert" };
      case "ensure_bridge_running":
        return new Promise<void>((res, rej) => {
          startGate = { open: res, fail: rej };
        });
      default:
        return undefined;
    }
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  _resetRestartBridgeForTests();
  record = { phase: "rebuilding", done: 3, total: 9 };
  mockInvoke.mockReset();
  fakeTauri();
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe("launchBrain: the start-up progress on the launch screen", () => {
  it("shows the progress while the blocking start runs, then returns to idle", async () => {
    const { result } = renderHook(() => useRestartBridge(P, "bridge_down"));
    let launch!: ReturnType<typeof launchBrain>;
    act(() => {
      launch = launchBrain(P);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000);
    });
    expect(result.current.state).toBe("starting");
    expect(result.current.progress).toBe(LINE);
    expect(result.current.offerHardExit).toBe(false); // the stop-only offers are not shown for a start
    startGate.open();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect((await launch).kind).toBe("ready");
    expect(result.current.state).toBe("idle");
    expect(result.current.progress).toBeNull();
    expect(restartInFlight()).toBe(false);
  });

  it("returns to idle when the start fails, and the error screen's result is unchanged", async () => {
    const { result } = renderHook(() => useRestartBridge(P, "bridge_down"));
    let launch!: ReturnType<typeof launchBrain>;
    act(() => {
      launch = launchBrain(P);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(7000);
    });
    expect(result.current.progress).toBe(LINE);
    startGate.fail("supervisor start failed (exit 1): persona directory not found");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(await launch).toEqual({
      kind: "error",
      error: "supervisor start failed (exit 1): persona directory not found",
      canUseReleaseBrain: false,
    });
    expect(result.current.state).toBe("idle");
    expect(result.current.progress).toBeNull();
    expect(restartInFlight()).toBe(false);
  });

  it("an ordinary quick start shows nothing and reads no status for it", async () => {
    const { result } = renderHook(() => useRestartBridge(P, "bridge_down"));
    let launch!: ReturnType<typeof launchBrain>;
    act(() => {
      launch = launchBrain(P);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000); // before the first 5 s read
    });
    const statusReads = () => mockInvoke.mock.calls.filter(([cmd]) => cmd === "bridge_status").length;
    const before = statusReads(); // the launch's own stop check reads once
    startGate.open();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect((await launch).kind).toBe("ready");
    expect(statusReads()).toBe(before);
    expect(result.current.progress).toBeNull();
  });
});
