// Vitest coverage for the bridge restart (#456 C17 a-m, Q-A, D7, D10, D18, D21, D22) and the one-restart-app-wide
// behaviour (#310, #335, #365).
//
// The bridge is a small in-memory "world": its liveness, stop phase, progress record and /health, as the app reads
// them through the mocked Tauri / HTTP calls. Time is fake: the 180 s offer and the 500 ms / 5 s polls are advanced.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor, cleanup } from "@testing-library/react";

import * as bridge from "../bridge";
import * as appConfig from "../appConfig";
import {
  _resetRestartBridgeForTests,
  awaitStoppingBridge,
  clearRestartState,
  EVERYTHING_STOPPED,
  HARD_EXIT_DONE,
  HARD_EXIT_OFFER_AFTER_MS,
  offerLine,
  progressLine,
  restartBridge,
  restartInFlight,
  TIMEOUT_SHUTDOWN_MS,
  useRestartBridge,
} from "./useRestartBridge";

const PERSONA = "test-persona";
const T0 = new Date("2026-10-19T21:00:00.000Z");

function jsonResponse(status: number, body: object = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface World {
  alive: "alive" | "gone";
  phase: "live" | "stopping";
  legacy: boolean;
  proven: boolean;
  /** /health answers (the listener closes once a stop begins). */
  health: boolean;
  /** A progress record of this bridge exists. */
  record: boolean;
  tasks: string[];
  requestedAt: string | null;
  /** The next N `status --json` reads time out ("unknown"). */
  unknownReads: number;
  statusUnknown: boolean;
  readError: boolean;
}

function fresh(): World {
  return {
    alive: "alive",
    phase: "live",
    legacy: false,
    proven: false,
    health: true,
    record: false,
    tasks: [],
    requestedAt: null,
    unknownReads: 0,
    statusUnknown: false,
    readError: false,
  };
}

let w: World = fresh();

/** The bridge begins its stop now: listener closed, record written, the request time recorded. */
function beginStop(tasks: string[] = w.tasks): void {
  w.phase = "stopping";
  w.health = false;
  w.record = true;
  w.tasks = tasks;
  w.requestedAt = new Date(Date.now()).toISOString();
}

function mocks() {
  return {
    status: vi.spyOn(bridge, "bridgeStatus").mockImplementation(async () => {
      if (w.unknownReads > 0) {
        w.unknownReads -= 1;
        return { alive: "unknown" };
      }
      if (w.statusUnknown) return { alive: "unknown" };
      if (w.alive === "gone") return { alive: "gone" };
      return {
        alive: "alive",
        legacy: w.legacy,
        proven: w.proven,
        phase: w.phase,
        stop_requested_at: w.requestedAt,
      };
    }),
    read: vi.spyOn(bridge, "readBridgeShutdown").mockImplementation(async () => {
      if (w.readError) throw new Error("read_bridge_shutdown: unreadable");
      if (w.alive !== "alive" || !w.record) return null;
      return { pid: 4242, pid_token: "tok", phase: "stopping", requested_at: w.requestedAt, tasks: [...w.tasks] };
    }),
    shutdown: vi.spyOn(bridge, "shutdownBridge").mockImplementation(async () => {
      if (!w.health) throw new TypeError("fetch failed");
      beginStop();
      return jsonResponse(202, { status: "stopping" });
    }),
    health: vi.spyOn(bridge, "fetchHealth").mockImplementation(async () => {
      if (w.alive === "alive" && w.health) return { liveness: "ok" } as never;
      throw new Error("/health unreachable");
    }),
    hardExit: vi.spyOn(bridge, "requestBridgeHardExit").mockResolvedValue(),
    force: vi.spyOn(bridge, "invokeForceRestart").mockImplementation(async () => {
      w = fresh(); // Rust: `stop --kill`, then the normal start
    }),
    start: vi.spyOn(appConfig, "ensureBridgeRunning").mockImplementation(async () => {
      w = fresh();
    }),
    snapshot: vi.spyOn(bridge, "snapshotActiveSession").mockResolvedValue(jsonResponse(200)),
  };
}

async function advance(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

function startRestart(restart: () => Promise<boolean>): { done: () => Promise<boolean> } {
  let p: Promise<boolean> = Promise.resolve(false);
  act(() => {
    p = restart();
  });
  return { done: () => p };
}

beforeEach(() => {
  vi.restoreAllMocks();
  _resetRestartBridgeForTests();
  w = fresh();
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe("useRestartBridge: a new bridge (#456)", () => {
  it("(a, b, c, e, i) asks for the stop, shows what runs, says Everything has stopped, then starts normally", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["chat reply", "background job: compaction"];
    const m = mocks();
    const { result, rerender } = renderHook(
      ({ mode }: { mode: "live" | "bridge_down" }) => useRestartBridge(PERSONA, mode),
      { initialProps: { mode: "bridge_down" } },
    );
    const run = startRestart(() => result.current.restart());
    await advance(1000);

    expect(m.shutdown).toHaveBeenCalledWith(PERSONA, undefined, "app_restart");
    expect(result.current.state).toBe("stopping");
    expect(result.current.progress).toBe(
      "Stopping. Waiting for 2 tasks to finish: chat reply, background job: compaction.",
    );

    w.tasks = ["background job: compaction"];
    await advance(1000);
    expect(result.current.progress).toBe(progressLine(["background job: compaction"]));
    expect(m.start).not.toHaveBeenCalled();

    w.alive = "gone"; // the last task ended and the bridge exited
    await advance(6000);
    expect(result.current.progress).toBe(EVERYTHING_STOPPED);
    expect(m.start).toHaveBeenCalledWith(PERSONA); // (i) the normal start, after the old bridge is gone
    await expect(run.done()).resolves.toBe(true);
    expect(result.current.state).toBe("reconnecting");

    expect(m.snapshot).not.toHaveBeenCalled(); // (a)
    expect(m.force).not.toHaveBeenCalled(); // (b) nothing writes a stop intent on a graceful stop
    expect(m.hardExit).not.toHaveBeenCalled();
    expect(result.current.offerHardExit).toBe(false);

    rerender({ mode: "live" });
    await waitFor(() => expect(result.current.state).toBe("success"));
  });

  it("(c) a 503 bridge_stopping means the stop already began", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    const m = mocks();
    m.shutdown.mockImplementation(async () => {
      beginStop(["soul review"]);
      return jsonResponse(503, { code: "bridge_stopping", message: "the bridge is stopping" });
    });
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    startRestart(() => result.current.restart());
    await advance(1000);
    expect(result.current.state).toBe("stopping");
    expect(m.shutdown).toHaveBeenCalledTimes(1);
    expect(m.force).not.toHaveBeenCalled();
    expect(result.current.progress).toBe(progressLine(["soul review"]));
    w.alive = "gone";
    await advance(6000);
    expect(m.start).toHaveBeenCalledTimes(1);
  });

  it("(d) while /health answers and no stop is recorded, the request is retried every 500 ms for 180 s, then fails without a kill", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    const m = mocks();
    m.shutdown.mockImplementation(async () => jsonResponse(500, { detail: "internal" }));
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());

    await advance(10_000);
    const after10s = m.shutdown.mock.calls.length;
    expect(after10s).toBeGreaterThanOrEqual(19);
    expect(after10s).toBeLessThanOrEqual(21);
    expect(result.current.state).toBe("requesting");

    await advance(HARD_EXIT_OFFER_AFTER_MS - 10_000 + 1000);
    await expect(run.done()).resolves.toBe(false);
    expect(result.current.state).toBe("failed");
    expect(result.current.errorDetail).toMatch(/Restart failed/);
    expect(m.shutdown.mock.calls.length).toBeLessThanOrEqual(HARD_EXIT_OFFER_AFTER_MS / 500 + 2);
    expect(m.force).not.toHaveBeenCalled();
    expect(m.start).not.toHaveBeenCalled();
  });

  it("(e) offers the hard exit at the recorded request + 180 s, kills nothing before acceptance, then asks the bridge", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["heartbeat (Claude call)"];
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());

    await advance(HARD_EXIT_OFFER_AFTER_MS - 1000);
    expect(result.current.offerHardExit).toBe(false);
    await advance(2000);
    expect(result.current.offerHardExit).toBe(true);
    expect(result.current.offerText).toBe(offerLine(["heartbeat (Claude call)"]));

    await advance(600_000); // no answer: the bridge keeps waiting, nothing is killed or asked
    expect(m.force).not.toHaveBeenCalled();
    expect(m.hardExit).not.toHaveBeenCalled();
    expect(result.current.state).toBe("stopping");

    act(() => result.current.hardExit());
    await advance(1000);
    expect(m.hardExit).toHaveBeenCalledWith(PERSONA);
    expect(result.current.offerHardExit).toBe(false);
    expect(m.force).not.toHaveBeenCalled();

    w.alive = "gone"; // the bridge acted on the request
    await advance(6000);
    expect(result.current.progress).toBe(HARD_EXIT_DONE);
    expect(m.start).toHaveBeenCalledWith(PERSONA);
    await expect(run.done()).resolves.toBe(true);
    expect(m.force).not.toHaveBeenCalled();
  });

  it("(e) Keep waiting hides the offer and never kills", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["soul review"];
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    startRestart(() => result.current.restart());
    await advance(HARD_EXIT_OFFER_AFTER_MS + 1000);
    expect(result.current.offerHardExit).toBe(true);
    act(() => result.current.keepWaiting());
    expect(result.current.offerHardExit).toBe(false);
    await advance(1_000_000);
    expect(result.current.offerHardExit).toBe(false); // offered once
    expect(m.hardExit).not.toHaveBeenCalled();
    expect(m.force).not.toHaveBeenCalled();
    expect(result.current.state).toBe("stopping");
    w.alive = "gone";
    await advance(6000);
    expect(result.current.progress).toBe(EVERYTHING_STOPPED);
  });

  it("(f) a status read that times out keeps waiting and a progress-record read error is retried", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["chat reply"];
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    startRestart(() => result.current.restart());
    await advance(1000);
    expect(result.current.state).toBe("stopping");

    w.statusUnknown = true;
    w.readError = true;
    const readsBefore = m.read.mock.calls.length;
    await advance(60_000);
    expect(m.read.mock.calls.length - readsBefore).toBeGreaterThanOrEqual(100); // retried every 500 ms
    expect(result.current.state).toBe("stopping");
    expect(m.start).not.toHaveBeenCalled();
    expect(m.force).not.toHaveBeenCalled();

    w.statusUnknown = false;
    w.readError = false;
    w.alive = "gone";
    await advance(6000);
    expect(m.start).toHaveBeenCalledTimes(1);
    expect(m.force).not.toHaveBeenCalled();
  });

  it("(g, m) kills only when /health and the request both fail and no stop is recorded", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.health = false; // hung: no /health, the request does not get through, nothing recorded
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(2000);
    expect(m.force).toHaveBeenCalledTimes(1);
    expect(m.force).toHaveBeenCalledWith(PERSONA, true); // (m) --skip-hard-exit-request
    expect(result.current.fellThrough).toMatch(/answered neither/);
    await expect(run.done()).resolves.toBe(true);
    expect(result.current.state).toBe("reconnecting");
  });

  it("(g) a failed request with a stop recorded is a stop in progress, not a kill", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["chat reply"];
    beginStop(); // another client asked first; the listener is closed
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    startRestart(() => result.current.restart());
    await advance(30_000);
    expect(m.force).not.toHaveBeenCalled();
    expect(result.current.state).toBe("stopping");
    expect(result.current.progress).toBe(progressLine(["chat reply"]));
    w.alive = "gone";
    await advance(6000);
    expect(m.start).toHaveBeenCalledTimes(1);
    expect(m.force).not.toHaveBeenCalled();
  });

  it("(m) a hard exit the bridge ignores for 180 s escalates to stop --kill --skip-hard-exit-request", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["heartbeat (Claude call)"];
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(HARD_EXIT_OFFER_AFTER_MS + 1000);
    act(() => result.current.hardExit());
    await advance(1000);
    expect(m.hardExit).toHaveBeenCalledTimes(1);

    await advance(HARD_EXIT_OFFER_AFTER_MS - 2000);
    expect(m.force).not.toHaveBeenCalled();
    await advance(3000);
    expect(m.force).toHaveBeenCalledWith(PERSONA, true);
    expect(result.current.fellThrough).toMatch(/did not act on the hard exit/);
    await expect(run.done()).resolves.toBe(true);
  });

  it("(m) a forced kill that errors (e.g. force_restart_timeout) shows the error and fails", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.health = false;
    const m = mocks();
    m.force.mockRejectedValue("force_restart_timeout: the bridge is still being stopped; try again in a moment");
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(2000);
    await expect(run.done()).resolves.toBe(false);
    expect(result.current.state).toBe("failed");
    expect(result.current.errorDetail).toMatch(/force_restart_timeout/);
  });
});

describe("useRestartBridge: a pre-PR bridge (#456 C17 j, k)", () => {
  it("(j) proven: the old request, then stop --kill 3 s later if still alive, no display, no offer, then the start", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = true;
    const m = mocks();
    m.shutdown.mockImplementation(async () => jsonResponse(202)); // accepts, keeps running
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(TIMEOUT_SHUTDOWN_MS - 100);
    expect(m.shutdown).toHaveBeenCalledWith(PERSONA);
    expect(m.force).not.toHaveBeenCalled();
    await advance(200);
    expect(m.force).toHaveBeenCalledWith(PERSONA, false);
    await expect(run.done()).resolves.toBe(true);
    expect(result.current.progress).toBeNull();
    expect(result.current.offerHardExit).toBe(false);
  });

  it("(j) proven and gone within 3 s: the normal start, no kill", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = true;
    const m = mocks();
    m.shutdown.mockImplementation(async () => {
      w.alive = "gone";
      return jsonResponse(202);
    });
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(TIMEOUT_SHUTDOWN_MS + 100);
    await expect(run.done()).resolves.toBe(true);
    expect(m.force).not.toHaveBeenCalled();
    expect(m.start).toHaveBeenCalledWith(PERSONA);
  });

  it("(j) an unproven pre-PR record takes the normal start at once", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = false;
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(100);
    await expect(run.done()).resolves.toBe(true);
    expect(m.start).toHaveBeenCalledWith(PERSONA);
    expect(m.shutdown).not.toHaveBeenCalled();
    expect(m.force).not.toHaveBeenCalled();
  });

  it("(k) an unknown first status is read again before deciding: a pre-PR bridge takes (j)", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = true;
    w.unknownReads = 1;
    const m = mocks();
    m.shutdown.mockImplementation(async () => jsonResponse(202));
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(TIMEOUT_SHUTDOWN_MS + 500);
    await expect(run.done()).resolves.toBe(true);
    expect(m.shutdown).toHaveBeenCalledTimes(1);
    expect(m.shutdown).toHaveBeenCalledWith(PERSONA); // the old request, not the new-bridge one
    expect(m.force).toHaveBeenCalledWith(PERSONA, false);
    expect(result.current.offerHardExit).toBe(false);
  });

  it("(k) a later status naming a pre-PR bridge switches the wait to (j)", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = true;
    w.unknownReads = 2; // both first reads time out
    const m = mocks();
    m.shutdown.mockImplementation(async () => jsonResponse(202)); // an old bridge also answers 202
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const run = startRestart(() => result.current.restart());
    await advance(TIMEOUT_SHUTDOWN_MS + 2000);
    await expect(run.done()).resolves.toBe(true);
    expect(m.force).toHaveBeenCalledWith(PERSONA, false);
    expect(m.force).toHaveBeenCalledTimes(1);
    expect(result.current.offerHardExit).toBe(false);
  });
});

describe("awaitStoppingBridge: the launch check (#456 C17 h, l)", () => {
  it("(h) a bridge found stopping: the same display, the offer at the recorded request + 180 s, no request sent", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.tasks = ["saving 1 conversation"];
    beginStop();
    w.requestedAt = new Date(T0.getTime() - 100_000).toISOString(); // asked 100 s ago
    const m = mocks();
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    let launched: Promise<boolean> = Promise.resolve(false);
    act(() => {
      launched = awaitStoppingBridge(PERSONA);
    });
    await advance(1000);
    expect(restartInFlight()).toBe(true);
    expect(result.current.state).toBe("stopping");
    expect(result.current.progress).toBe(progressLine(["saving 1 conversation"]));
    await advance(78_000);
    expect(result.current.offerHardExit).toBe(false);
    await advance(2000); // 80 s after the launch check
    expect(result.current.offerHardExit).toBe(true);
    expect(m.shutdown).not.toHaveBeenCalled();

    w.alive = "gone";
    await advance(6000);
    await expect(launched).resolves.toBe(true);
    expect(result.current.progress).toBe(EVERYTHING_STOPPED);
    expect(m.start).not.toHaveBeenCalled(); // the caller starts (brainRecovery.launchBrain)
    expect(m.force).not.toHaveBeenCalled();
    expect(restartInFlight()).toBe(false);
  });

  it("(l) a proven pre-PR bridge whose /health is closed is killed", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = true;
    w.health = false;
    const m = mocks();
    let launched: Promise<boolean> = Promise.resolve(false);
    act(() => {
      launched = awaitStoppingBridge(PERSONA);
    });
    await advance(1000);
    await expect(launched).resolves.toBe(true);
    expect(m.force).toHaveBeenCalledWith(PERSONA, false);
  });

  it("(l) a pre-PR bridge that still answers, or a new bridge not stopping, is left alone", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(T0);
    w.legacy = true;
    w.proven = true;
    const m = mocks();
    await act(async () => {
      await expect(awaitStoppingBridge(PERSONA)).resolves.toBe(true);
    });
    w = fresh();
    await act(async () => {
      await expect(awaitStoppingBridge(PERSONA)).resolves.toBe(true);
    });
    expect(m.force).not.toHaveBeenCalled();
    expect(m.shutdown).not.toHaveBeenCalled();
    expect(m.read).not.toHaveBeenCalled();
  });
});

describe("useRestartBridge: one restart app-wide (#310, #335)", () => {
  /** Hold the stop request open until released; on release the bridge exits at once. */
  function holdRequest(m: ReturnType<typeof mocks>): () => void {
    let release!: () => void;
    m.shutdown.mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          release = () => {
            w.alive = "gone";
            resolve(jsonResponse(202));
          };
        }),
    );
    return () => release();
  }

  it("re-entry guard: clicking restart while in flight sends one request", async () => {
    const m = mocks();
    const release = holdRequest(m);
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    let first: Promise<boolean> = Promise.resolve(false);
    act(() => {
      first = result.current.restart();
      result.current.restart();
      result.current.restart();
    });
    await waitFor(() => expect(m.shutdown).toHaveBeenCalledTimes(1));
    await act(async () => {
      release();
      await first;
    });
    expect(m.shutdown).toHaveBeenCalledTimes(1);
  });

  it("a second restart() while one is in flight joins it and resolves with its result", async () => {
    const m = mocks();
    const release = holdRequest(m);
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    let first: Promise<boolean> = Promise.resolve(false);
    let second: Promise<boolean> = Promise.resolve(false);
    act(() => {
      first = result.current.restart();
      second = result.current.restart();
    });
    await waitFor(() => expect(m.shutdown).toHaveBeenCalledTimes(1));
    await act(async () => {
      release();
      await first;
    });
    expect(await second).toBe(true);
    expect(m.start).toHaveBeenCalledTimes(1);
  });

  it("(#310) separate hook instances share one restart and one state", async () => {
    const m = mocks();
    const release = holdRequest(m);
    const a = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    const b = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    let fromA: Promise<boolean> = Promise.resolve(false);
    act(() => {
      fromA = a.result.current.restart();
    });
    await waitFor(() => expect(b.result.current.state).toBe("requesting"));
    let fromB: Promise<boolean> = Promise.resolve(false);
    act(() => {
      fromB = b.result.current.restart();
    });
    await act(async () => {
      release();
      await fromA;
    });
    expect(await fromB).toBe(true);
    expect(m.shutdown).toHaveBeenCalledTimes(1);
    expect(b.result.current.state).toBe("reconnecting");
  });

  it("(#335) restartBridge() outside React drives the same shared restart", async () => {
    const m = mocks();
    const release = holdRequest(m);
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    let fromLaunch: Promise<boolean> = Promise.resolve(false);
    act(() => {
      fromLaunch = restartBridge(PERSONA);
    });
    await waitFor(() => expect(result.current.state).toBe("requesting"));
    let fromHook: Promise<boolean> = Promise.resolve(false);
    act(() => {
      fromHook = result.current.restart();
    });
    await act(async () => {
      release();
      await fromLaunch;
    });
    expect(await fromHook).toBe(true);
    expect(m.shutdown).toHaveBeenCalledTimes(1);
  });

  it("(#335) clearRestartState() hides a restart nobody pressed, but never one in flight", async () => {
    const m = mocks();
    const release = holdRequest(m);
    const { result } = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    let pending: Promise<boolean> = Promise.resolve(false);
    act(() => {
      pending = restartBridge(PERSONA);
    });
    await waitFor(() => expect(result.current.state).toBe("requesting"));
    act(() => clearRestartState());
    expect(result.current.state).toBe("requesting"); // in flight: untouched
    await act(async () => {
      release();
      await pending;
    });
    expect(result.current.state).toBe("reconnecting");
    act(() => clearRestartState());
    expect(result.current.state).toBe("idle");
    expect(m.start).toHaveBeenCalledTimes(1);
  });

  it("(#310) a shared success goes back to idle when the bridge drops again", async () => {
    const m = mocks();
    m.shutdown.mockImplementation(async () => {
      w.alive = "gone";
      return jsonResponse(202);
    });
    const { result, rerender } = renderHook(
      ({ mode }: { mode: "live" | "bridge_down" }) => useRestartBridge(PERSONA, mode),
      { initialProps: { mode: "bridge_down" } },
    );
    await act(async () => {
      await result.current.restart();
    });
    rerender({ mode: "live" });
    await waitFor(() => expect(result.current.state).toBe("success"));
    rerender({ mode: "bridge_down" });
    await waitFor(() => expect(result.current.state).toBe("idle"));
    const freshHook = renderHook(() => useRestartBridge(PERSONA, "bridge_down"));
    expect(freshHook.result.current.state).toBe("idle");
  });
});
