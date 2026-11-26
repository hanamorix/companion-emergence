// #456 C24: the app's crash restart. A crashed bridge (gone, no recorded stop, no committed kill of it, not a
// service's) is started again after 30 s, 60 s, 300 s, then every 600 s, never giving up; 600 s healthy (or a
// recorded stop) resets the backoff. Fake timers; the status, /health and start are injected.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import type { BridgeStatus } from "../bridge";
import { BridgeWatchdog, CRASH_RESTART_DELAYS_MS, HEALTHY_RESET_MS, type WatchdogDeps } from "./useBridgeWatchdog";

const PERSONA = "test-persona";
const TOKEN = "linux:boot:4242:1";

function crashed(over: Partial<BridgeStatus> = {}): BridgeStatus {
  return {
    alive: "gone",
    phase: "running",
    pid: 4242,
    pid_token: TOKEN,
    stop_reason: null,
    stop_intent: null,
    client_origin: "tauri",
    ...over,
  };
}

let status: BridgeStatus;
let up: boolean;
let startBrings: boolean;
let busy: boolean;
let deps: WatchdogDeps;
let dog: BridgeWatchdog;

beforeEach(() => {
  vi.useFakeTimers();
  status = crashed();
  up = false;
  startBrings = false;
  busy = false;
  deps = {
    status: vi.fn(async () => status),
    healthy: vi.fn(async () => up),
    start: vi.fn(async () => {
      if (startBrings) {
        up = true;
        status = { ...status, alive: "alive" };
      }
    }),
    busy: () => busy,
  };
  dog = new BridgeWatchdog(PERSONA, deps);
});

afterEach(() => {
  dog.stop();
  vi.useRealTimers();
});

async function advance(ms: number): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
}

describe("BridgeWatchdog (#456 C24)", () => {
  it("restarts a crash after 30 s, 60 s, 300 s, then every 600 s, never giving up", async () => {
    dog.onMode("bridge_down");
    const expected = [30_000, 60_000, 300_000, 600_000, 600_000, 600_000, 600_000];
    for (let i = 0; i < expected.length; i++) {
      await advance(expected[i] - 1);
      expect(deps.start).toHaveBeenCalledTimes(i);
      await advance(1);
      expect(deps.start).toHaveBeenCalledTimes(i + 1);
    }
    expect(CRASH_RESTART_DELAYS_MS).toEqual([30_000, 60_000, 300_000, 600_000]);
  });

  it("600 s healthy after a restart resets the backoff to 30 s; a crash before that continues it", async () => {
    startBrings = true;
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(1);
    dog.onMode("live");

    // A crash 100 s later (not yet 600 s healthy): the next wait is 60 s.
    await advance(100_000);
    up = false;
    status = crashed();
    dog.onMode("bridge_down");
    await advance(59_999);
    expect(deps.start).toHaveBeenCalledTimes(1);
    await advance(1);
    expect(deps.start).toHaveBeenCalledTimes(2);

    // Healthy for the full 600 s: the next crash waits 30 s again.
    dog.onMode("live");
    await advance(HEALTHY_RESET_MS);
    up = false;
    status = crashed();
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(3);
  });

  it("a recorded stop also ends the incident: the next crash waits 30 s", async () => {
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(1); // failed; the next wait would be 60 s
    status = crashed({ phase: "stopped", stop_reason: "user_stop" });
    dog.onMode("live");
    dog.onMode("bridge_down");
    await advance(60_000);
    expect(deps.start).toHaveBeenCalledTimes(1); // a stop is not a crash
    status = crashed();
    dog.onMode("live");
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(2);
  });

  const neverRestarted: [string, BridgeStatus][] = [
    ["an idle or user stop", crashed({ phase: "stopped", stop_reason: "idle" })],
    ["a stop still recorded as stopping (hard exit)", crashed({ phase: "stopping", stop_reason: "user_stop" })],
    [
      "a user kill of this bridge (committed intent)",
      crashed({ stop_intent: { state: "committed", pid: 4242, pid_token: TOKEN, reason: "user_kill" } }),
    ],
    ["a launchd service bridge", crashed({ client_origin: "launchd" })],
    ["a systemd service bridge", crashed({ client_origin: "systemd" })],
    ["a Windows task bridge", crashed({ client_origin: "task-scheduler" })],
  ];
  it.each(neverRestarted)("never restarts %s", async (_label, s) => {
    status = s;
    dog.onMode("bridge_down");
    await advance(2_000_000);
    expect(deps.start).not.toHaveBeenCalled();
  });

  it("(review S7) a run the bridge itself recorded as a crash is restarted", async () => {
    status = crashed({ phase: "hard_exit", stop_reason: "crash" });
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(1);
  });

  it("a pending intent (a kill that never happened) or another bridge's committed intent does not stop it", async () => {
    status = crashed({ stop_intent: { state: "pending", pid: 4242, pid_token: TOKEN, reason: "user_kill" } });
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(1);

    dog.stop();
    dog = new BridgeWatchdog(PERSONA, deps);
    status = crashed({ stop_intent: { state: "committed", pid: 4242, pid_token: "linux:boot:4242:0" } });
    dog.onMode("bridge_down");
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(2);
  });

  it("stays out of an app restart or brain update in flight, and looks again once it is over", async () => {
    busy = true;
    dog.onMode("bridge_down");
    await advance(120_000);
    expect(deps.start).not.toHaveBeenCalled();
    expect(deps.status).not.toHaveBeenCalled();
    busy = false;
    await advance(10_000); // the next status read finds the crash
    await advance(30_000);
    expect(deps.start).toHaveBeenCalledTimes(1);
  });

  it("re-checks /health and the record right before each attempt", async () => {
    dog.onMode("bridge_down");
    await advance(20_000);
    up = true; // it came back on its own
    await advance(10_000);
    expect(deps.start).not.toHaveBeenCalled();

    up = false;
    dog.onMode("live");
    dog.onMode("bridge_down");
    await advance(20_000);
    status = crashed({ phase: "stopped", stop_reason: "user_stop" }); // stopped meanwhile
    await advance(10_000);
    expect(deps.start).not.toHaveBeenCalled();
  });

  it("a status read that times out is read again, and a crash found later is restarted", async () => {
    status = { alive: "unknown" };
    dog.onMode("bridge_down");
    await advance(5_000);
    expect(deps.status).toHaveBeenCalledTimes(2); // at once, then after STATUS_POLL_MS
    status = crashed(); // then it is gone without a recorded stop
    await advance(30_000); // the next read (the ladder's 30 s step) finds the crash
    expect(deps.status).toHaveBeenCalledTimes(3);
    await advance(29_999);
    expect(deps.start).not.toHaveBeenCalled();
    await advance(1);
    expect(deps.start).toHaveBeenCalledTimes(1);
  });

  it("(review S6) a recorded stop is not read again", async () => {
    status = { ...crashed(), alive: "alive", phase: "stopping", stop_reason: "user_stop" };
    dog.onMode("bridge_down");
    await advance(3_600_000);
    expect(deps.status).toHaveBeenCalledTimes(1);
    expect(deps.start).not.toHaveBeenCalled();
  });

  it("(review S6) an inconclusive read backs off: 5 s, then 30 s, 1 min, 5 min, then every 10 min", async () => {
    status = { alive: "unknown" };
    dog.onMode("bridge_down");
    const readsAt = async (t: number, n: number) => {
      await advance(t - Date.now());
      expect(deps.status).toHaveBeenCalledTimes(n);
    };
    const t0 = Date.now();
    vi.mocked(deps.status).mockClear();
    // the first read already happened at t0
    await readsAt(t0 + 4_999, 0);
    await readsAt(t0 + 5_000, 1);
    await readsAt(t0 + 34_999, 1);
    await readsAt(t0 + 35_000, 2);
    await readsAt(t0 + 95_000, 3);
    await readsAt(t0 + 395_000, 4);
    await readsAt(t0 + 995_000, 5);
    await readsAt(t0 + 1_595_000, 6);
    expect(deps.start).not.toHaveBeenCalled();
  });

  it("(review S6) the back-off starts again when the reading changes", async () => {
    status = { alive: "unknown" };
    dog.onMode("bridge_down");
    const t0 = Date.now();
    await advance(395_000); // reads at 0, 5 s, 35 s, 95 s, 395 s; the next one is due 10 min later
    vi.mocked(deps.status).mockClear();
    status = { ...crashed(), alive: "alive", phase: "running" }; // now the bridge answers status but not /health
    await advance(t0 + 995_000 - Date.now()); // the read that sees the change
    expect(deps.status).toHaveBeenCalledTimes(1);
    await advance(5_000); // from the start of the ladder again
    expect(deps.status).toHaveBeenCalledTimes(2);
  });

  it("stop() cancels a pending restart", async () => {
    dog.onMode("bridge_down");
    await advance(10_000);
    dog.stop();
    await advance(1_000_000);
    expect(deps.start).not.toHaveBeenCalled();
  });
});
