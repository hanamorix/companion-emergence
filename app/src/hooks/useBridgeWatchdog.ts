/**
 * useBridgeWatchdog: restart a bridge that crashed (#456 Q4, D7, D25; C24).
 *
 * A crash is a bridge that is gone while its record still says "running", with no recorded stop (no stop reason), no
 * COMMITTED stop intent from a user kill of that same bridge (a pending one is a kill that never happened), not run by
 * an OS service (the service manager restarts those), and no restart or brain update in flight. The watchdog starts
 * the bridge again after 30 s, then 60 s, 300 s, then every 600 s, never giving up (owner, 2026-10-19: "first time
 * restart after 30 seconds, second time 1min, then 5min, then 10min repeating. backoff timer resets on success").
 * Success (D7) = healthy, then 600 s healthy (or a recorded stop): the next crash starts again at 30 s. The status
 * and /health are read again right before each attempt. While the bridge is unreachable and the status cannot yet
 * call it a crash (the bridge is still there, or the status read timed out), or a restart is in flight, the status
 * is read again every STATUS_POLL_MS; a bridge gone after a recorded stop is left alone.
 */

import { useEffect } from "react";
import type { BridgeStatus, PersonaState } from "../bridge";
import { bridgeStatus, fetchHealth } from "../bridge";
import { ensureBridgeRunning } from "../appConfig";
import { restartInFlight, STATUS_POLL_MS } from "./useRestartBridge";
import { brainUpdateInFlight } from "./useBrainUpdate";

/** Owner Q4 (2026-10-19): 30 s, 1 min, 5 min, then 10 min repeating. */
export const CRASH_RESTART_DELAYS_MS = [30_000, 60_000, 300_000, 600_000] as const;
/** D7 (main-accepted): a restart counts as a success once the bridge has been healthy this long. */
export const HEALTHY_RESET_MS = 600_000;

const SERVICE_ORIGINS = new Set(["launchd", "systemd", "task-scheduler"]);

/** The crash test (C24) on one `nell supervisor status --json` reading. */
export function isCrash(s: BridgeStatus): boolean {
  if (s.alive !== "gone") return false;
  if (s.phase !== "running") return false; // a stop that was recorded (stopping, stopped, hard exit) is not a crash
  if (s.stop_reason) return false;
  const intent = s.stop_intent;
  if (intent && intent.state === "committed" && intent.pid === s.pid && intent.pid_token === s.pid_token) {
    return false; // a user kill of this bridge
  }
  if (s.client_origin && SERVICE_ORIGINS.has(s.client_origin)) return false;
  return true;
}

export interface WatchdogDeps {
  status: (persona: string) => Promise<BridgeStatus>;
  healthy: (persona: string) => Promise<boolean>;
  start: (persona: string) => Promise<void>;
  busy: () => boolean;
}

const defaultDeps: WatchdogDeps = {
  status: async (persona) => {
    try {
      return await bridgeStatus(persona);
    } catch {
      return { alive: "unknown" }; // read again later
    }
  },
  healthy: async (persona) => {
    try {
      await fetchHealth(persona);
      return true;
    } catch {
      return false;
    }
  },
  start: (persona) => ensureBridgeRunning(persona),
  busy: () => restartInFlight() || brainUpdateInFlight(),
};

/** The watchdog for one persona. `onMode` is fed the app's /state mode; `stop` cancels everything. */
export class BridgeWatchdog {
  private attempts = 0;
  private attemptTimer: ReturnType<typeof setTimeout> | null = null;
  private resetTimer: ReturnType<typeof setTimeout> | null = null;
  private recheckTimer: ReturnType<typeof setTimeout> | null = null;
  private mode: PersonaState["mode"] | null = null;
  private checking = false;
  private stopped = false;

  constructor(
    private readonly persona: string,
    private readonly deps: WatchdogDeps = defaultDeps,
  ) {}

  get attemptCount(): number {
    return this.attempts;
  }

  onMode(mode: PersonaState["mode"]): void {
    if (this.stopped) return;
    this.mode = mode;
    if (mode === "live") {
      this.clearRecheck();
      if (this.attempts > 0 && this.resetTimer === null) {
        this.resetTimer = setTimeout(() => {
          this.resetTimer = null;
          this.attempts = 0; // D7: healthy for 600 s, the backoff starts again at 30 s
        }, HEALTHY_RESET_MS);
      }
      return;
    }
    if (this.resetTimer !== null) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }
    void this.check();
  }

  stop(): void {
    this.stopped = true;
    if (this.attemptTimer !== null) clearTimeout(this.attemptTimer);
    if (this.resetTimer !== null) clearTimeout(this.resetTimer);
    this.clearRecheck();
    this.attemptTimer = null;
    this.resetTimer = null;
  }

  private clearRecheck(): void {
    if (this.recheckTimer !== null) clearTimeout(this.recheckTimer);
    this.recheckTimer = null;
  }

  /** Read the status again later while the bridge stays unreachable. */
  private recheckLater(): void {
    if (this.stopped || this.recheckTimer !== null || this.mode !== "bridge_down") return;
    this.recheckTimer = setTimeout(() => {
      this.recheckTimer = null;
      if (this.mode === "bridge_down") void this.check();
    }, STATUS_POLL_MS);
  }

  private async check(): Promise<void> {
    if (this.checking || this.attemptTimer !== null) return;
    this.checking = true;
    try {
      if (this.deps.busy()) {
        this.recheckLater(); // a restart or brain update in flight: look again once it is over
        return;
      }
      const s = await this.deps.status(this.persona);
      if (!isCrash(s)) {
        if (s.stop_reason) this.attempts = 0; // a recorded stop also ends the incident (D7)
        if (s.alive !== "gone") this.recheckLater(); // still there (hung or stopping), or the read timed out
        return;
      }
      const delay = CRASH_RESTART_DELAYS_MS[Math.min(this.attempts, CRASH_RESTART_DELAYS_MS.length - 1)];
      this.attemptTimer = setTimeout(() => void this.attempt(), delay);
    } finally {
      this.checking = false;
    }
  }

  private async attempt(): Promise<void> {
    this.attemptTimer = null;
    if (this.stopped) return;
    if (this.deps.busy()) {
      this.recheckLater();
      return;
    }
    // Re-check right before the attempt: a bridge that came back, or a stop recorded meanwhile, is left alone.
    if (await this.deps.healthy(this.persona)) return;
    const s = await this.deps.status(this.persona);
    if (!isCrash(s)) {
      if (s.stop_reason) this.attempts = 0; // a recorded stop ends the incident (D7)
      if (s.alive !== "gone") this.recheckLater();
      return;
    }
    this.attempts += 1;
    try {
      await this.deps.start(this.persona);
    } catch (e) {
      console.warn("[bridgeWatchdog] restart after a crash failed; trying again later", e);
    }
    if (!(await this.deps.healthy(this.persona))) void this.check(); // still down: the next, longer wait
  }
}

/** Mount once for the selected persona: feeds the app's /state mode to the watchdog. */
export function useBridgeWatchdog(persona: string, currentMode: PersonaState["mode"] | undefined): void {
  const key = persona;
  useEffect(() => {
    const dog = new BridgeWatchdog(key);
    watchdogs.set(key, dog);
    return () => {
      dog.stop();
      watchdogs.delete(key);
    };
  }, [key]);
  useEffect(() => {
    if (currentMode !== undefined) watchdogs.get(key)?.onMode(currentMode);
  }, [key, currentMode]);
}

const watchdogs = new Map<string, BridgeWatchdog>();
