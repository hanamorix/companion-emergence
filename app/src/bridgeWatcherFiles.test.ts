/**
 * #456 C48: the app reads the same golden watcher files the Python writers produce
 * (tests/fixtures/bridge_watcher_files/, checked against the writers by tests/unit/brain/bridge/test_watcher_file_formats.py).
 * Each fixture is parsed here and fed to the TS code that reads that shape.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import type { BridgeShutdown, BridgeStatus } from "./bridge";
import { isCrash } from "./hooks/useBridgeWatchdog";
import { offerLine, progressLine } from "./hooks/useRestartBridge";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "tests", "fixtures", "bridge_watcher_files");

function fixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURES, name), "utf8")) as T;
}

describe("golden watcher files (#456 C48)", () => {
  it("progress.json: the stop display reads its tasks and request time", () => {
    const rec = fixture<BridgeShutdown>("progress.json");
    expect(typeof rec.pid).toBe("number");
    expect(typeof rec.pid_token).toBe("string");
    expect(rec.phase).toBe("stopping");
    expect(Number.isNaN(Date.parse(rec.requested_at ?? ""))).toBe(false);
    expect(progressLine(rec.tasks)).toBe(
      "Stopping. Waiting for 2 tasks to finish: background job: compaction, chat reply.",
    );
    expect(offerLine(rec.tasks)).toBe(
      "Still waiting for 2 tasks after 3 minutes: background job: compaction, chat reply. Hard exit now? " +
        "The next start will pick up anything left unfinished.",
    );
  });

  it("hard_exit_request.json: the keys the app's request writer must produce", () => {
    const req = fixture<Record<string, unknown>>("hard_exit_request.json");
    expect(Object.keys(req).sort()).toEqual(["pid", "pid_token", "requested_at", "version"]);
    expect(req.version).toBe(1);
  });

  it("status.json: the restart flow and the watchdog read alive, phase, the intent with its state, and the bound", () => {
    const s = fixture<BridgeStatus>("status.json");
    expect(["alive", "gone", "unknown"]).toContain(s.alive);
    expect(s.phase).toBe("stopping");
    expect(s.stop_intent?.state).toBe("pending");
    expect(s.remaining?.tasks).toEqual(["background job: compaction", "chat reply"]);
    expect(s.supervisor?.state).toBe("running");
    expect(typeof s.kill_bound_s).toBe("number");
    expect(isCrash(s)).toBe(false); // alive
  });

  it("stop intents: only a committed intent of the same bridge suppresses a crash restart", () => {
    const base = fixture<BridgeStatus>("status.json");
    const committed = fixture<NonNullable<BridgeStatus["stop_intent"]>>("stop_intent_committed.json");
    const pending = fixture<NonNullable<BridgeStatus["stop_intent"]>>("stop_intent_pending.json");
    const gone: BridgeStatus = { ...base, alive: "gone", phase: "running", stop_reason: null, client_origin: "tauri" };
    expect(isCrash({ ...gone, stop_intent: committed })).toBe(false);
    expect(isCrash({ ...gone, stop_intent: pending })).toBe(true);
    expect(isCrash({ ...gone, stop_intent: null })).toBe(true);
    expect(isCrash({ ...gone, stop_intent: { ...committed, pid_token: "another-bridge" } })).toBe(true);
  });
});
