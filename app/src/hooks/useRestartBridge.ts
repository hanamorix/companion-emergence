/**
 * useRestartBridge: the app's bridge restart (#456 Q-A, D7, D10, D21, D22).
 *
 * A restart asks the bridge to stop and waits, with no time limit of its own, until everything it was doing has
 * finished: the user sees what is still running, and 180 s after the stop was requested is offered a hard exit
 * (owner Q1). Only then is the new bridge started. Nothing is killed without the user's say-so, except a bridge
 * that answers neither its /health nor the stop request and has no stop recorded (the forced path), and a pre-PR
 * bridge (one too old to report its stop), which keeps today's short wait before the kill.
 *
 *   idle -> requesting -> stopping -> starting -> waiting_for_health -> reconnecting -> success
 *                     \-> forcing (no answer at all, or a hard exit the bridge ignored) -> waiting_for_health ...
 *   any step -> failed
 *
 * The bridge's HTTP listener closes once a stop begins, so the stop is followed through files only: the progress
 * record (`read_bridge_shutdown`) and `nell supervisor status --json` (`bridge_status`).
 *
 * One restart app-wide (#310): App, RestartBridgeButton and useBrainUpdate each hold an instance, so the state and
 * the in-flight run live at module scope. A caller that asks while a restart is running joins it.
 */

import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { BridgeShutdown, BridgeStatus, PersonaState } from "../bridge";
import {
  bridgeStatus,
  fetchHealth,
  invokeForceRestart,
  readBridgeShutdown,
  requestBridgeHardExit,
  shutdownBridge,
} from "../bridge";
import { ensureBridgeRunning } from "../appConfig";
import { errString } from "../lib/errString";
import { HEALTH_POLL_INTERVAL_MS, HEALTH_WAIT_MS, REQUEST_TIMEOUT_MS } from "../lib/requestTimeout";

export type RestartState =
  | "idle"
  | "requesting"
  | "stopping"
  | "forcing"
  | "starting"
  | "waiting_for_health"
  | "reconnecting"
  | "success"
  | "failed";

export interface UseRestartBridge {
  state: RestartState;
  errorDetail: string | null;
  /** Why the restart had to force the old bridge down, else null (#365). */
  fellThrough: string | null;
  /** The stop's progress line while stopping (the same words the terminal shows), else null. */
  progress: string | null;
  /** True while the hard-exit offer is shown (180 s after the stop was requested). */
  offerHardExit: boolean;
  /** The offer's text while it is shown, else null. */
  offerText: string | null;
  restart: () => Promise<boolean>;
  hardExit: () => void;
  keepWaiting: () => void;
  onModeChanged: (mode: PersonaState["mode"]) => void;
}

/** The pre-PR request budget before its kill (unchanged from the earlier restart flow). Also the time limit on each
 *  single request this flow sends (the stop request, each /health probe): a wedged bridge can accept the connection
 *  and never answer (review M1). */
export const TIMEOUT_SHUTDOWN_MS = REQUEST_TIMEOUT_MS;
export const TIMEOUT_HEALTH_MS = HEALTH_WAIT_MS;
export { HEALTH_POLL_INTERVAL_MS };
/** Owner Q1 (2026-10-19): the hard exit is offered 180 s after the stop was requested (brain/bridge/stop_watch.py
 *  HARD_EXIT_OFFER_AFTER_S). Also how long a bridge that still answers /health gets for the stop request to go
 *  through, and how long an accepted hard exit gets before the kill. */
export const HARD_EXIT_OFFER_AFTER_MS = 180_000;
/** How often the stop is re-read while waiting (progress record) and the status (`nell supervisor status`). */
export const PROGRESS_POLL_MS = 500;
export const STATUS_POLL_MS = 5000;

export const EVERYTHING_STOPPED = "Everything has stopped.";
export const HARD_EXIT_DONE = "Hard exit done. The next start will pick up anything left unfinished.";

function plural(n: number): string {
  return n === 1 ? "task" : "tasks";
}

export function progressLine(tasks: string[]): string {
  if (tasks.length === 0) return "Stopping. Finishing up.";
  return `Stopping. Waiting for ${tasks.length} ${plural(tasks.length)} to finish: ${tasks.join(", ")}.`;
}

export function offerLine(tasks: string[]): string {
  const minutes = Math.floor(HARD_EXIT_OFFER_AFTER_MS / 60_000);
  return (
    `Still waiting for ${tasks.length} ${plural(tasks.length)} after ${minutes} minutes: ${tasks.join(", ")}. ` +
    "Hard exit now? The next start will pick up anything left unfinished."
  );
}

const FAILED_USER_MESSAGE =
  "Restart failed. Try `nell service status` from terminal, or restart Companion Emergence.";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Run one request with a time limit: past ``ms`` it is aborted and the call rejects (review M1). */
export async function withRequestTimeout<T>(run: (signal: AbortSignal) => Promise<T>, ms: number): Promise<T> {
  const ac = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expired = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      ac.abort();
      reject(new Error(`no answer within ${ms} ms`));
    }, ms);
  });
  try {
    return await Promise.race([run(ac.signal), expired]);
  } finally {
    clearTimeout(timer);
  }
}

/** One /health probe, bounded by TIMEOUT_SHUTDOWN_MS. */
export function probeHealth(persona: string): Promise<unknown> {
  return withRequestTimeout((signal) => fetchHealth(persona, signal), TIMEOUT_SHUTDOWN_MS);
}

async function pollHealth(persona: string, deadline: number): Promise<void> {
  while (Date.now() < deadline) {
    try {
      await probeHealth(persona);
      return;
    } catch {
      await sleep(HEALTH_POLL_INTERVAL_MS);
    }
  }
  throw new Error("/health poll window expired");
}

async function healthAnswers(persona: string): Promise<boolean> {
  try {
    await probeHealth(persona);
    return true;
  } catch {
    return false;
  }
}

interface Shared {
  state: RestartState;
  errorDetail: string | null;
  fellThrough: string | null;
  progress: string | null;
  offerHardExit: boolean;
  offerText: string | null;
}

const IDLE: Shared = {
  state: "idle",
  errorDetail: null,
  fellThrough: null,
  progress: null,
  offerHardExit: false,
  offerText: null,
};

let shared: Shared = { ...IDLE };
let inFlight: Promise<boolean> | null = null;
/** The user's answer to the hard-exit offer for the stop being watched. */
let answer: "hard_exit" | "keep_waiting" | null = null;
const listeners = new Set<() => void>();

function setShared(patch: Partial<Shared>): void {
  shared = { ...shared, ...patch };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function transition(next: RestartState): void {
  setShared({ state: next });
}

/** True while a restart (or a launch-time wait for a stopping bridge) runs. The crash watchdog stays out of it. */
export function restartInFlight(): boolean {
  return inFlight !== null;
}

/** Back to idle after a restart nobody pressed (the launch path, #335). A no-op while a restart runs. */
export function clearRestartState(): void {
  if (!inFlight) setShared({ ...IDLE });
}

/** Reset for test isolation; never call in production code. */
export function _resetRestartBridgeForTests(): void {
  shared = { ...IDLE };
  inFlight = null;
  answer = null;
}

/** The user accepted the hard-exit offer. */
export function acceptHardExit(): void {
  if (shared.offerHardExit) {
    answer = "hard_exit";
    setShared({ offerHardExit: false, offerText: null });
  }
}

/** The user chose to keep waiting (the offer is shown once). */
export function keepWaitingForStop(): void {
  if (shared.offerHardExit) {
    answer = "keep_waiting";
    setShared({ offerHardExit: false, offerText: null });
  }
}

function requestedAtMs(rec: BridgeShutdown | null, status: BridgeStatus | null, fallback: number): number {
  for (const value of [rec?.requested_at, status?.stop_requested_at]) {
    if (value) {
      const t = Date.parse(value);
      if (!Number.isNaN(t)) return t;
    }
  }
  return fallback;
}

type StopOutcome = "gone" | "legacy" | "forced" | "failed";

/**
 * Watch a stop that has begun until the bridge is gone (C17 e, f, h): the progress line, the hard-exit offer at the
 * recorded request time + 180 s, the hard-exit request on acceptance, and the kill if the bridge ignores it for 180 s.
 * A status read that times out ("unknown") keeps waiting; a progress-record read error is retried.
 */
/** The bridge a stop watch follows: its pid + identity token (review S8). */
interface Pin {
  pid: number;
  token: string | null;
}

function pinOf(s: BridgeStatus): Pin | null {
  return s.alive === "alive" && typeof s.pid === "number" ? { pid: s.pid, token: s.pid_token ?? null } : null;
}

function isPinned(pin: Pin, pid: number | null | undefined, token: string | null | undefined): boolean {
  return pin.pid === pid && pin.token === (token ?? null);
}

async function watchStop(persona: string, pinned: Pin | null, start = true): Promise<StopOutcome> {
  transition("stopping");
  answer = null;
  const watchStarted = Date.now();
  let pin = pinned;
  let status: BridgeStatus | null = null;
  let rec: BridgeShutdown | null = null;
  let lastStatusAt = -Infinity;
  let offered = false;
  let acceptedAt: number | null = null;
  const gone = (): StopOutcome => {
    setShared({ progress: acceptedAt !== null ? HARD_EXIT_DONE : EVERYTHING_STOPPED, offerHardExit: false,
      offerText: null });
    return "gone";
  };
  setShared({ progress: "Stopping." }); // until the first read says what is still running
  for (;;) {
    if (Date.now() - lastStatusAt >= STATUS_POLL_MS) {
      lastStatusAt = Date.now();
      status = await bridgeStatus(persona);
      if (status.alive === "gone") return gone();
      if (status.alive === "alive") {
        // S8: the watched bridge has ended and another one runs in its place: the watch is over.
        if (pin !== null && !isPinned(pin, status.pid, status.pid_token)) return gone();
        pin = pin ?? pinOf(status);
        if (status.legacy) return "legacy";
      }
    }
    try {
      const next = await readBridgeShutdown(persona);
      if (next) {
        if (pin !== null && !isPinned(pin, next.pid, next.pid_token)) return gone(); // a record of another bridge
        pin = pin ?? { pid: next.pid, token: next.pid_token ?? null };
        rec = next;
      }
    } catch {
      // retried at the next poll (f)
    }
    if (rec) setShared({ progress: progressLine(rec.tasks ?? []) });
    const now = Date.now();
    if (!offered && now >= requestedAtMs(rec, status, watchStarted) + HARD_EXIT_OFFER_AFTER_MS) {
      offered = true;
      setShared({ offerHardExit: true, offerText: offerLine(rec?.tasks ?? []) });
    }
    if (answer === "hard_exit" && acceptedAt === null) {
      acceptedAt = now;
      try {
        await requestBridgeHardExit(persona, pin ? { pid: pin.pid, pid_token: pin.token } : undefined);
      } catch (e) {
        if (errString(e).startsWith("bridge_changed")) return gone(); // never a request to a bridge not asked about
        console.warn(`[restartBridge] hard-exit request failed: ${errString(e)}`);
      }
    }
    if (acceptedAt !== null && now >= acceptedAt + HARD_EXIT_OFFER_AFTER_MS) {
      // (m) the bridge ignored the hard exit: the identity-checked kill, without asking again, and only while the
      // watched bridge is still the one running (S8).
      const latest = await bridgeStatus(persona);
      if (latest.alive === "gone" || (pin !== null && latest.alive === "alive" && !isPinned(pin, latest.pid, latest.pid_token))) {
        return gone();
      }
      setShared({ fellThrough: "the bridge did not act on the hard exit" });
      return (await forced(persona, true, start, pin)) ? "forced" : "failed";
    }
    await sleep(PROGRESS_POLL_MS);
  }
}

/** The forced path: `nell supervisor stop --kill [--skip-hard-exit-request]`, then (``start``) the normal start, in
 *  Rust. Without ``start`` (a service install, M4) the bridge is only ended. */
async function forced(persona: string, skipHardExitRequest: boolean, start = true, pin: Pin | null = null): Promise<boolean> {
  transition("forcing");
  try {
    if (pin) {
      await invokeForceRestart(persona, skipHardExitRequest, start, { pid: pin.pid, pid_token: pin.token });
    } else {
      await invokeForceRestart(persona, skipHardExitRequest, start); // a pre-PR bridge has no token to pin
    }
  } catch (e) {
    if (errString(e).includes("bridge_changed")) {
      // Review S8: the watched bridge is already gone and another one runs; nothing was killed.
      return start ? startFresh(persona) : true;
    }
    setShared({ errorDetail: errString(e) || FAILED_USER_MESSAGE });
    return false;
  }
  if (!start) return true;
  transition("waiting_for_health");
  try {
    await pollHealth(persona, Date.now() + TIMEOUT_HEALTH_MS);
  } catch {
    setShared({ errorDetail: FAILED_USER_MESSAGE });
    return false;
  }
  return true;
}

/** Wait for a bridge something else is starting (the service an install started) to answer /health. */
async function awaitHealth(persona: string): Promise<boolean> {
  transition("waiting_for_health");
  try {
    await pollHealth(persona, Date.now() + TIMEOUT_HEALTH_MS);
  } catch {
    setShared({ errorDetail: FAILED_USER_MESSAGE });
    return false;
  }
  return true;
}

async function startFresh(persona: string): Promise<boolean> {
  transition("starting");
  try {
    await ensureBridgeRunning(persona);
  } catch (e) {
    setShared({ errorDetail: errString(e) || FAILED_USER_MESSAGE });
    return false;
  }
  transition("waiting_for_health");
  try {
    await pollHealth(persona, Date.now() + TIMEOUT_HEALTH_MS);
  } catch {
    setShared({ errorDetail: FAILED_USER_MESSAGE });
    return false;
  }
  return true;
}

/** A pre-PR bridge (j, l): the old request, today's short wait, then the kill (with its tree) and (``start``) the
 *  start. */
async function stopPrePr(persona: string, start = true): Promise<boolean> {
  transition("requesting");
  try {
    await withRequestTimeout((signal) => shutdownBridge(persona, signal), TIMEOUT_SHUTDOWN_MS);
  } catch {
    // its listener may be closed already, or it did not answer in time
  }
  await sleep(TIMEOUT_SHUTDOWN_MS);
  const after = await bridgeStatus(persona);
  if (after.alive === "alive") return forced(persona, false, start);
  return start ? startFresh(persona) : true;
}

async function readStatusSettled(persona: string): Promise<BridgeStatus> {
  // (k) an "unknown" first read is read again before anything is decided.
  const first = await bridgeStatus(persona);
  if (first.alive !== "unknown") return first;
  return bridgeStatus(persona);
}

function stopBegan(r: Response): Promise<boolean> {
  if (r.status === 202) return Promise.resolve(true);
  if (r.status !== 503) return Promise.resolve(false);
  return r
    .json()
    .then((body: { code?: string; detail?: { code?: string } }) =>
      body?.code === "bridge_stopping" || body?.detail?.code === "bridge_stopping")
    .catch(() => false);
}

async function stopIsRecorded(persona: string): Promise<boolean> {
  try {
    if (await readBridgeShutdown(persona)) return true;
  } catch {
    // fall through to the status
  }
  const s = await bridgeStatus(persona);
  return s.alive !== "gone" && s.phase === "stopping";
}

/** Ask a new bridge to stop (C17 c, d, g). "stopping" once it began; "forced" after the kill path; "failed". */
async function requestStop(
  persona: string,
  start = true,
  pin: Pin | null = null,
): Promise<"stopping" | "forced" | "failed"> {
  transition("requesting");
  const deadline = Date.now() + HARD_EXIT_OFFER_AFTER_MS;
  for (;;) {
    try {
      const r = await withRequestTimeout(
        (signal) => shutdownBridge(persona, signal, "app_restart"),
        TIMEOUT_SHUTDOWN_MS,
      );
      if (await stopBegan(r)) return "stopping";
    } catch {
      // the request did not get through; decide below
    }
    if (await stopIsRecorded(persona)) return "stopping";
    if (!(await healthAnswers(persona))) {
      // (g) neither /health nor the request answers and no stop is recorded: the forced path
      setShared({ fellThrough: "the bridge answered neither /health nor the stop request" });
      return (await forced(persona, true, start, pin)) ? "forced" : "failed";
    }
    if (Date.now() >= deadline) {
      setShared({ errorDetail: FAILED_USER_MESSAGE });
      return "failed";
    }
    await sleep(HEALTH_POLL_INTERVAL_MS);
  }
}

/**
 * Launch check (C17 h, l): before starting, a bridge found stopping is waited for with the same display and offer;
 * a proven pre-PR bridge whose /health is closed is killed. Resolves when the persona has no bridge in the way.
 */
export function awaitStoppingBridge(persona: string): Promise<boolean> {
  if (inFlight) return inFlight;
  const run = async (): Promise<boolean> => {
    try {
      const status = await readStatusSettled(persona);
      if (status.alive !== "alive") return true;
      if (status.legacy) {
        if (!status.proven || (await healthAnswers(persona))) return true;
        return forced(persona, false);
      }
      if (status.phase !== "stopping") return true;
      const outcome = await watchStop(persona, pinOf(status));
      if (outcome === "legacy") return stopPrePr(persona);
      return outcome !== "failed";
    } finally {
      inFlight = null;
    }
  };
  inFlight = run();
  return inFlight;
}

/** The app-wide restart (#310). Plain function so code that runs before any component mounts can call it. */
export function restartBridge(persona: string): Promise<boolean> {
  if (inFlight) return inFlight;
  setShared({ ...IDLE });

  const run = async (): Promise<boolean> => {
    try {
      const ok = await stopBridge(persona, true);
      if (!ok) {
        transition("failed");
        return false;
      }
      transition("reconnecting");
      return true;
    } finally {
      inFlight = null;
    }
  };

  inFlight = run();
  return inFlight;
}

/**
 * Stop the persona's bridge through the same stop flow (display, offer, forced path), and (``start``) start the next
 * one. True once the old bridge is gone (and, with ``start``, the new one answers).
 */
async function stopBridge(persona: string, start: boolean): Promise<boolean> {
  const finish = (): Promise<boolean> => (start ? startFresh(persona) : Promise.resolve(true));
  const status = await readStatusSettled(persona);
  if (status.alive === "gone") return finish();
  if (status.legacy) {
    // (j) a pre-PR bridge: proven -> the old request and the kill after 3 s; unproven -> the normal start.
    return status.proven ? stopPrePr(persona, start) : finish();
  }
  const pin = pinOf(status);
  const requested = await requestStop(persona, start, pin);
  if (requested === "forced") return true;
  if (requested === "failed") return false;
  const outcome = await watchStop(persona, pin, start);
  if (outcome === "forced") return true;
  if (outcome === "failed") return false;
  if (outcome === "legacy") return stopPrePr(persona, start);
  return finish(); // (i) "Everything has stopped." -> the normal start
}

export interface StopThenResult {
  /** The old bridge was stopped (the action ran). */
  stopped: boolean;
  /** Why the action, the stop or the start failed; null when all went well. */
  error: string | null;
}

/**
 * C58 (d), review M4: the app's service install. The bridge is stopped through the stop flow first (the install
 * refuses a running bridge), then ``action`` runs on a stopped bridge, then (``start``) the normal start, which goes
 * through the newly installed service. Waits for a restart already running. Shares the restart's state and display.
 */
export async function stopBridgeThen(
  persona: string,
  action: () => Promise<void>,
  { start }: { start: boolean },
): Promise<StopThenResult> {
  while (inFlight) await inFlight.catch(() => false);
  setShared({ ...IDLE });
  const result: StopThenResult = { stopped: false, error: null };
  const run = async (): Promise<boolean> => {
    try {
      if (!(await stopBridge(persona, false))) {
        result.error = shared.errorDetail ?? FAILED_USER_MESSAGE;
        transition("failed");
        return false;
      }
      result.stopped = true;
      try {
        await action();
      } catch (e) {
        result.error = errString(e) || "failed";
      }
      if (start) {
        // Review R2-M1: a successful install already started the service, so wait for its bridge instead of a
        // second start (which would find the service's start under way); after a failed one, the normal start.
        const back = result.error === null ? await awaitHealth(persona) : await startFresh(persona);
        if (!back) {
          result.error = result.error ?? shared.errorDetail ?? FAILED_USER_MESSAGE;
          transition("failed");
          return false;
        }
        transition("reconnecting");
      }
      return result.error === null;
    } finally {
      inFlight = null;
    }
  };
  inFlight = run();
  await inFlight;
  if (!start) clearRestartState();
  return result;
}

export function useRestartBridge(
  persona: string,
  currentMode: PersonaState["mode"],
): UseRestartBridge {
  const { state, errorDetail, fellThrough, progress, offerHardExit, offerText } = useSyncExternalStore(
    subscribe,
    () => shared,
  );

  const restart = useCallback(() => restartBridge(persona), [persona]);

  const onModeChanged = useCallback((mode: PersonaState["mode"]) => {
    if (shared.state === "reconnecting" && mode === "live") {
      transition("success");
    }
  }, []);

  useEffect(() => {
    if (state === "reconnecting" && currentMode === "live") {
      transition("success");
    } else if (state === "success" && currentMode !== "live") {
      transition("idle");
    }
  }, [currentMode, state]);

  return {
    state,
    errorDetail,
    fellThrough,
    progress,
    offerHardExit,
    offerText,
    restart,
    hardExit: acceptHardExit,
    keepWaiting: keepWaitingForStop,
    onModeChanged,
  };
}
