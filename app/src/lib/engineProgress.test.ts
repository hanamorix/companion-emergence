// The app's wait for a starting bridge, paused while the engine's start-up record shows (HR5 part 4, TS half).
// `bridgeStatus` is mocked; time is fake. Budget 30 s, status read at most every 5 s, /health tried every 500 ms.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("../bridge", () => ({ bridgeStatus: vi.fn() }));

import { bridgeStatus } from "../bridge";
import { alongside, waitForHealth } from "./engineProgress";
import { HEALTH_POLL_INTERVAL_MS, HEALTH_WAIT_MS, STATUS_POLL_MS } from "./requestTimeout";

const REC = { phase: "rebuilding", done: 1, total: 2 };
const SHOWN = { alive: "alive", engine_progress: REC } as const;
const ABSENT = { alive: "alive", engine_progress: null } as const;
const TIMED_OUT = { alive: "unknown" } as const;

const status = vi.mocked(bridgeStatus);
let next: unknown;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const down = async (): Promise<never> => {
  throw new Error("/health unreachable");
};

function startWait(probe: () => Promise<unknown> = down, onProgress?: (r: unknown) => void) {
  const settled = vi.fn();
  const p = waitForHealth("p", probe, HEALTH_WAIT_MS, onProgress as never);
  p.then(settled, settled);
  return { p, settled };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-11-16T12:00:00.000Z"));
  status.mockReset();
  next = ABSENT;
  status.mockImplementation(async () => next as never);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("waitForHealth", () => {
  it("answers true at once when /health answers, with no status read", async () => {
    const probe = vi.fn(async () => ({ liveness: "ok" }));
    const { p } = startWait(probe);
    await expect(p).resolves.toBe(true);
    expect(status).not.toHaveBeenCalled();
  });

  it("gives up at its budget when no engine progress shows", async () => {
    const { p, settled } = startWait();
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS - 2000);
    expect(settled).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(3000 + STATUS_POLL_MS);
    await expect(p).resolves.toBe(false);
  });

  it("is paused while a reading shows engine progress, then gives up at the budget once it is absent", async () => {
    next = SHOWN;
    const { p, settled } = startWait();
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS * 3);
    expect(settled).not.toHaveBeenCalled(); // 90 s in, past three budgets
    next = ABSENT;
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS - 10_000);
    expect(settled).not.toHaveBeenCalled(); // the budget starts again from what is left, not from zero minus the pause
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS + STATUS_POLL_MS);
    await expect(p).resolves.toBe(false);
  });

  it("a status read that times out keeps the previous reading for under 30 s after the last real answer, then the wait gives up", async () => {
    // Real reads at 0, 5 and 10 s (the last real answer is at 10 s); every read from 12 s on times out. The pause holds
    // while the reading is under 30 s old (the last credited tick is 39.5 s, so the deadline stops at 30 + 39.5 = 69.5 s
    // and the wait gives up at the 70 s tick).
    next = SHOWN;
    const { settled } = startWait();
    await vi.advanceTimersByTimeAsync(12_000);
    next = TIMED_OUT;
    await vi.advanceTimersByTimeAsync(27_000); // 39 s
    expect(settled).not.toHaveBeenCalled(); // under 30 s after the last real answer: still paused
    await vi.advanceTimersByTimeAsync(21_000); // 60 s
    expect(settled).not.toHaveBeenCalled(); // the deadline has stopped moving, what is left of its limit is not yet used up
    await vi.advanceTimersByTimeAsync(15_000); // 75 s
    expect(settled).toHaveBeenCalledWith(false);
  });

  it("a real reading that takes 4 s ages from its answer, and later timed-out reads do not move it", async () => {
    // The first read is asked at 0 s and answers at 4 s with a record; every later read times out at once. The age runs
    // from 4 s: the last credited tick is 33.5 s, the deadline stops at 63.5 s. Aged from the read's start (0 s) the wait
    // would give up at about 60 s; if a timed-out read moved the age it would never give up.
    let calls = 0;
    status.mockImplementation(async () => {
      calls += 1;
      if (calls === 1) {
        await sleep(4000);
        return SHOWN as never;
      }
      return TIMED_OUT as never;
    });
    const { settled } = startWait();
    await vi.advanceTimersByTimeAsync(61_000);
    expect(calls).toBeGreaterThan(5); // later reads did happen and timed out
    expect(settled).not.toHaveBeenCalled(); // not aged from the read's start
    await vi.advanceTimersByTimeAsync(4000); // 65 s
    expect(settled).toHaveBeenCalledWith(false); // and not kept alive by the timed-out reads
  });

  it("a fresh real reading restores a lapsed pause", async () => {
    // Real to 12 s, timed out from 12 s: the pause lapses at the 40 s tick and the deadline stops at 69.5 s. A record is
    // read again at 55 s (set at 54 s) while the wait is still alive; reads time out again from 57 s. The last real answer
    // is then 55 s, the credit runs to 84.5 s and the wait gives up at the 100 s tick (without the restore, at about 70 s).
    next = SHOWN;
    const { settled } = startWait();
    await vi.advanceTimersByTimeAsync(12_000);
    next = TIMED_OUT;
    await vi.advanceTimersByTimeAsync(42_000); // 54 s
    expect(settled).not.toHaveBeenCalled();
    next = SHOWN;
    await vi.advanceTimersByTimeAsync(3000); // 57 s: the read at 55 s was real
    next = TIMED_OUT;
    await vi.advanceTimersByTimeAsync(33_000); // 90 s
    expect(settled).not.toHaveBeenCalled(); // the restored pause is still holding past the 70 s of the lapsed one
    await vi.advanceTimersByTimeAsync(15_000); // 105 s
    expect(settled).toHaveBeenCalledWith(false);
  });

  it("a timed-out read after an absent one stays absent: the wait still ends at its budget", async () => {
    next = ABSENT;
    const { p } = startWait();
    await vi.advanceTimersByTimeAsync(5000);
    next = TIMED_OUT;
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS + STATUS_POLL_MS);
    await expect(p).resolves.toBe(false);
  });

  it("a status of null or undefined counts as a timed-out read, not a crash", async () => {
    next = undefined;
    const { p } = startWait();
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS + STATUS_POLL_MS);
    await expect(p).resolves.toBe(false);
    expect(status).toHaveBeenCalled(); // it did ask
  });

  it("a status with no engine_progress field (an older daemon) is absent", async () => {
    next = { alive: "alive" };
    const { p } = startWait();
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS + STATUS_POLL_MS);
    await expect(p).resolves.toBe(false);
  });

  it("reads the status every STATUS_POLL_MS, not on the 500 ms health loop", async () => {
    const probe = vi.fn(down);
    startWait(probe);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(probe.mock.calls.length).toBeGreaterThanOrEqual(20_000 / HEALTH_POLL_INTERVAL_MS - 2); // the loop ran at 500 ms
    expect(status.mock.calls.length).toBeGreaterThanOrEqual(4); // a read at 0, 5, 10, 15 s
    expect(status.mock.calls.length).toBeLessThanOrEqual(5);
  });

  it("passes the one reading it used to onProgress (null when absent), once per read", async () => {
    const seen: unknown[] = [];
    next = SHOWN;
    startWait(down, (r) => seen.push(r));
    await vi.advanceTimersByTimeAsync(6000);
    expect(seen.length).toBeGreaterThanOrEqual(1);
    expect(seen[0]).toEqual(REC);
    next = ABSENT;
    await vi.advanceTimersByTimeAsync(STATUS_POLL_MS + 1000);
    expect(seen[seen.length - 1]).toBeNull();
    expect(seen.length).toBe(status.mock.calls.length);
  });

  it("with no callback it still pauses, and asks only for the status", async () => {
    next = SHOWN;
    const { settled } = startWait(down, undefined);
    await vi.advanceTimersByTimeAsync(HEALTH_WAIT_MS * 2);
    expect(settled).not.toHaveBeenCalled();
    expect(status).toHaveBeenCalled();
  });

  it("takes one more read before giving up, so a record that appeared since the last read is seen", async () => {
    // Each loop is 1200 ms of probe + 500 ms of waiting, so the ticks fall at 1200 + 1700k ms: the due reads land at 1.2, 6.3,
    // 11.4, 16.5, 21.6 and 26.7 s, and the first tick past the 30 s deadline (30.1 s) is not due for a read of its own.
    const slowDown = async (): Promise<never> => {
      await sleep(1200);
      throw new Error("down");
    };
    const start = Date.now();
    const reads: number[] = [];
    status.mockImplementation(async () => {
      reads.push(Date.now() - start);
      // the record appears at 28 s, after the last due read (26.7 s) and before the give-up tick
      return (Date.now() - start >= 28_000 ? SHOWN : ABSENT) as never;
    });
    const settled = vi.fn();
    const p = waitForHealth("p", slowDown, HEALTH_WAIT_MS);
    p.then(settled, settled);
    await vi.advanceTimersByTimeAsync(30_500);
    expect(reads[reads.length - 1]).toBeGreaterThanOrEqual(30_000); // the extra read, at the tick that would have given up
    expect(settled).not.toHaveBeenCalled(); // it saw the record, so the wait goes on
  });
});

describe("alongside: the status loop beside a blocking call", () => {
  function deferred() {
    let open!: (v: string) => void;
    let fail!: (e: Error) => void;
    const promise = new Promise<string>((res, rej) => {
      open = res;
      fail = rej;
    });
    return { promise, open, fail };
  }

  it("reads the status every STATUS_POLL_MS while the call is pending and passes each reading on", async () => {
    next = SHOWN;
    const work = deferred();
    const seen: unknown[] = [];
    const p = alongside("p", (r) => seen.push(r), () => work.promise);
    await vi.advanceTimersByTimeAsync(12_000);
    expect(status).toHaveBeenCalledTimes(2); // 5 s and 10 s: a start that ends sooner costs no extra read
    expect(seen).toEqual([REC, REC]);
    work.open("done");
    await expect(p).resolves.toBe("done");
    expect(vi.getTimerCount()).toBe(0); // the pending 5 s step is cancelled
    await vi.advanceTimersByTimeAsync(20_000);
    expect(status).toHaveBeenCalledTimes(2);
  });

  it("with no callback it does no status read at all and passes the result through", async () => {
    const work = deferred();
    const p = alongside("p", undefined, () => work.promise);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(status).not.toHaveBeenCalled();
    work.open("ok");
    await expect(p).resolves.toBe("ok");
  });

  it("lets the call's error through and ends the loop", async () => {
    next = SHOWN;
    const work = deferred();
    const p = alongside("p", () => undefined, () => work.promise);
    const caught = p.catch((e) => e);
    await vi.advanceTimersByTimeAsync(6000);
    expect(status).toHaveBeenCalledTimes(1); // the loop was running
    work.fail(new Error("exit 2"));
    expect(((await caught) as Error).message).toBe("exit 2");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("a read that returns after the call settled calls nothing", async () => {
    let finishRead!: (v: unknown) => void;
    status.mockImplementation(() => new Promise((res) => (finishRead = res)) as never);
    const work = deferred();
    const onProgress = vi.fn();
    const p = alongside("p", onProgress, () => work.promise);
    await vi.advanceTimersByTimeAsync(STATUS_POLL_MS + 100); // the read is out
    expect(status).toHaveBeenCalledTimes(1);
    work.open("done");
    await p;
    finishRead(SHOWN);
    await vi.advanceTimersByTimeAsync(100);
    expect(onProgress).not.toHaveBeenCalled(); // so endEngineProgress is the last word
  });
});
