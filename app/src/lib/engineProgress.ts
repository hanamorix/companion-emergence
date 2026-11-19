/**
 * The app's wait for a starting bridge, and the status read that follows it (HR5 part 4, TS half; owner S101, S102).
 *
 * While a bridge starts, the engine may run its first rebuild or the scale update before the bridge answers /health.
 * The daemon reports that as `engine_progress` in `nell supervisor status --json` (only while the process that writes
 * the record is alive). The app's 30 s waits for /health (`pollHealth`, `awaitHealth`, `awaitStartedBridge`) pause
 * their budget while it shows and run it as before otherwise: "the rebuild takes as long as it takes".
 *
 * One reader serves every wait here. It asks `bridgeStatus` at most once per STATUS_POLL_MS (not on the 500 ms health
 * loop), uses that one reading for the deadline, and hands it to an optional `onProgress` callback for the display.
 * A status read that timed out (`alive: "unknown"`, or no status at all) keeps the previous reading on screen, and keeps
 * the pause for less than the wait's own budget after the last real answer arrived; then the budget runs again with what
 * is left of it. A status without the field counts as absent. Time before the record appears counts against the budget.
 *
 * Imports only `bridgeStatus` (and a type) from `../bridge`, so appConfig.ts and useRestartBridge.ts can both import it
 * without a cycle.
 */

import type { EngineProgress } from "../bridge";
import { bridgeStatus } from "../bridge";
import { HEALTH_POLL_INTERVAL_MS, STATUS_POLL_MS } from "./requestTimeout";

/** Receives each reading: the record while one shows, null once it is absent. */
export type EngineProgressCallback = (reading: EngineProgress | null) => void;

interface Reader {
  /** The latest reading (the previous one after a timed-out read, which the display keeps showing). */
  reading: EngineProgress | null;
  /** True while the latest read was a real answer (not timed out). */
  real: boolean;
  /** When the last real answer ARRIVED (not when its read started); never moved by a timed-out read. */
  lastRealAt: number;
  /** True once STATUS_POLL_MS has passed since the last read started. */
  due(): boolean;
  /** Asks for the status and passes the reading to the callback (not after `stop`). */
  read(): Promise<void>;
  /** No further callback: a read still out when the wait ended must not show anything. */
  stop(): void;
}

function statusReader(persona: string, onProgress: EngineProgressCallback | undefined, firstReadAfterPoll: boolean): Reader {
  let lastReadAt = firstReadAfterPoll ? Date.now() : Number.NEGATIVE_INFINITY;
  let stopped = false;
  const reader: Reader = {
    reading: null,
    real: false,
    lastRealAt: Number.NEGATIVE_INFINITY,
    due: () => Date.now() - lastReadAt >= STATUS_POLL_MS,
    async read() {
      lastReadAt = Date.now();
      const status = await bridgeStatus(persona);
      if (stopped) return;
      // `alive: "unknown"` is a read that timed out or failed; a missing status is the same. Either keeps the reading
      // on screen, but it is not a new answer: `lastRealAt` is the time of the last real one, taken now that it came back.
      if (status && status.alive !== "unknown") {
        reader.reading = status.engine_progress ?? null;
        reader.real = true;
        reader.lastRealAt = Date.now();
      } else {
        reader.real = false;
      }
      onProgress?.(reader.reading);
    },
    stop() {
      stopped = true;
    },
  };
  return reader;
}

/**
 * Wait for `probe` (a /health request) to succeed, trying every HEALTH_POLL_INTERVAL_MS. True when it does; false once
 * `budgetMs` has passed with no engine progress showing. While a real reading shows progress (or a timed-out read keeps
 * one for less than `budgetMs` after the last real answer) the deadline moves out by the time that passes, so the wait
 * ends only after the record is gone or the readings stopped arriving, and the budget has run out. The first status read
 * comes with the first failed probe (a start that answers at once reads nothing); before giving up it reads once more,
 * so a record that appeared since the last read is seen.
 */
export async function waitForHealth(
  persona: string,
  probe: () => Promise<unknown>,
  budgetMs: number,
  onProgress?: EngineProgressCallback,
): Promise<boolean> {
  const reader = statusReader(persona, onProgress, false);
  let deadline = Date.now() + budgetMs;
  let lastTickAt = Date.now();
  for (;;) {
    try {
      await probe();
      return true;
    } catch {
      // not answering yet
    }
    let readThisTick = false;
    if (reader.due()) {
      await reader.read();
      readThisTick = true;
    }
    let now = Date.now();
    if (now >= deadline && !readThisTick) {
      await reader.read();
      now = Date.now();
    }
    // Pause while the reading is a real one, or was kept from a timed-out read for less than this wait's own budget after
    // the last real answer arrived; past that the deadline stops moving and the give-up clock runs with what is left of it.
    // Accepted edge (P4, S109): a real rebuild whose status reads time out for over 30 s can end the wait early. Owner:
    // "most status displays I've seen just keep the last reading until the next one is available or the thing crashes. so
    // go with the 30 seconds option." The display still shows the last reading.
    if (reader.reading !== null && (reader.real || now - reader.lastRealAt < budgetMs)) deadline += now - lastTickAt;
    lastTickAt = now;
    if (now >= deadline) return false;
    await new Promise<void>((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
  }
}

/**
 * Run a blocking call (`ensure_bridge_running`, the forced restart's kill and start) and, while it is pending, read the
 * status every STATUS_POLL_MS and pass each reading to `onProgress`. The first read comes after one interval, so a start
 * that ends sooner costs no status read. Ends with the call, whatever its outcome (the result or error passes through);
 * a read still out then calls nothing. With no callback there is nothing to show and no status is read.
 */
export async function alongside<T>(
  persona: string,
  onProgress: EngineProgressCallback | undefined,
  work: () => Promise<T>,
): Promise<T> {
  if (!onProgress) return work();
  const reader = statusReader(persona, onProgress, true);
  const running = work();
  let settled = false;
  const done = running.then(
    () => undefined,
    () => undefined,
  ).then(() => {
    settled = true;
  });
  try {
    while (!settled) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const step = new Promise<void>((r) => {
        timer = setTimeout(r, STATUS_POLL_MS);
      });
      await Promise.race([done, step]);
      clearTimeout(timer);
      if (settled) break;
      await Promise.race([reader.read(), done]);
    }
  } finally {
    reader.stop();
  }
  return running;
}
