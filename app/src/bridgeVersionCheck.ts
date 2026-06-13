/**
 * bridgeVersionCheck.ts — boot-time stale-bridge detection (V2 handshake).
 *
 * After `ensureBridgeRunning`, compares /health.version against the Tauri
 * app version.  On a mismatch: ONE automatic force-replace (invokeForceRestart
 * — kill-by-pid + reap + ensure, blocks until healthy).  Never loops — a
 * module-level flag guards the second attempt.
 *
 * Returns:
 *   "ok"                         — versions match, nothing to do
 *   "restarted"                  — mismatch detected, replace succeeded
 *   "version_mismatch_unresolved"— mismatch persists after replace (or flag
 *                                  already set from a prior call this session)
 *   "skipped"                    — dev build / unparseable version / fetchHealth
 *                                  threw; boot continues normally (never breaks
 *                                  on a hiccup)
 */

import { getVersion } from "@tauri-apps/api/app";
import { fetchHealth, invokeForceRestart } from "./bridge";

export type BridgeVersionResult =
  | "ok"
  | "restarted"
  | "version_mismatch_unresolved"
  | "skipped";

/** True after one replace attempt has been made this session. */
let _restartAttempted = false;

/** Reset for test isolation — never call in production code. */
export function _resetForTests(): void {
  _restartAttempted = false;
}

/**
 * Parse a semver string into a [major, minor, patch] triple.
 * Accepts trailing components (e.g. "0.0.34.0") by anchoring only the first
 * three numeric segments.  Returns null on missing/unparseable input.
 */
export function _parseSemver(
  v: string | undefined | null,
): [number, number, number] | null {
  if (!v) return null;
  const m = /^(\d+)\.(\d+)\.(\d+)/.exec(v.trim());
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

export async function ensureBridgeCurrent(
  persona: string,
): Promise<BridgeVersionResult> {
  // 1. Resolve Tauri app version — if unavailable (browser dev / error), skip.
  let appVersion: string;
  try {
    appVersion = await getVersion();
  } catch {
    return "skipped";
  }

  // 1B. Parse app version — dev build ("0.0.0" / starts with "0.0.0") or
  //     unparseable → never restart (would destructively replace a running bridge
  //     in dev mode).
  const appV = _parseSemver(appVersion);
  if (appV === null || appVersion.startsWith("0.0.0")) {
    return "skipped";
  }

  // 2. Probe the bridge — if it errors, skip (boot must not break on this).
  let health: { liveness: string; version?: string };
  try {
    health = await fetchHealth(persona);
  } catch {
    return "skipped";
  }

  // 3. Compare versions (semver-normalised).
  //    - health.version present but garbage → skip (don't destructively restart
  //      on a bridge we can't identify).
  //    - health.version missing/empty → treat as MISMATCH (pre-handshake bridge
  //      that predates the version field — intended restart).
  //    - both parse → ok if equal triples, else MISMATCH.
  if (health.version) {
    // Non-empty string present — try to parse.
    const healthV = _parseSemver(health.version);
    if (healthV === null) {
      // Present-but-garbage: skip rather than destroy an unknown bridge.
      return "skipped";
    }
    // Both parsed — compare triples.
    if (
      appV[0] === healthV[0] &&
      appV[1] === healthV[1] &&
      appV[2] === healthV[2]
    ) {
      return "ok";
    }
    // Explicit version mismatch → fall through to replace.
  }
  // health.version missing/empty → fall through to replace.

  // 4. Mismatch path — only one attempt per session.
  if (_restartAttempted) {
    return "version_mismatch_unresolved";
  }
  _restartAttempted = true;

  console.warn(
    `[bridgeVersionCheck] stale bridge detected (bridge=${health.version ?? "unknown"}, app=${appVersion}) — replacing`,
  );

  // 5. Force-replace: Tauri command kills by pid, reaps, re-ensures, blocks
  //    until healthy.  Shutdown + ensureBridgeRunning can't replace a gracefully-
  //    draining bridge (the old bridge answers /health 200 for up to 30s).
  try {
    await invokeForceRestart(persona);
  } catch {
    return "version_mismatch_unresolved";
  }

  // 6. Recheck (normalised compare — same logic as step 3).
  try {
    const recheckHealth = await fetchHealth(persona);
    if (recheckHealth.version) {
      const recheckV = _parseSemver(recheckHealth.version);
      if (
        recheckV !== null &&
        appV[0] === recheckV[0] &&
        appV[1] === recheckV[1] &&
        appV[2] === recheckV[2]
      ) {
        return "restarted";
      }
    }
  } catch {
    // fetchHealth failing after restart is still unresolved.
  }

  return "version_mismatch_unresolved";
}
