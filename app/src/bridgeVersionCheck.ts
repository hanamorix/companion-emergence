/**
 * bridgeVersionCheck.ts — boot-time stale-bridge detection (V2 handshake).
 *
 * After `ensureBridgeRunning`, compares /health.version against the Tauri
 * app version.  On a mismatch: ONE automatic replace (shutdown → ensure →
 * recheck).  Never loops — a module-level flag guards the second attempt.
 *
 * Returns:
 *   "ok"                         — versions match, nothing to do
 *   "restarted"                  — mismatch detected, replace succeeded
 *   "version_mismatch_unresolved"— mismatch persists after replace (or flag
 *                                  already set from a prior call this session)
 *   "skipped"                    — getVersion or fetchHealth threw; boot
 *                                  continues normally (never breaks on a hiccup)
 */

import { getVersion } from "@tauri-apps/api/app";
import { fetchHealth, shutdownBridge } from "./bridge";
import { ensureBridgeRunning } from "./appConfig";

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

  // 2. Probe the bridge — if it errors, skip (boot must not break on this).
  let health: { liveness: string; version?: string };
  try {
    health = await fetchHealth(persona);
  } catch {
    return "skipped";
  }

  // 3. Compare versions.  Missing version field == pre-handshake bridge → mismatch.
  if (health.version === appVersion) {
    return "ok";
  }

  // 4. Mismatch path — only one attempt per session.
  if (_restartAttempted) {
    return "version_mismatch_unresolved";
  }
  _restartAttempted = true;

  console.warn(
    `[bridgeVersionCheck] stale bridge detected (bridge=${health.version ?? "unknown"}, app=${appVersion}) — replacing`,
  );

  // 5. Attempt replace: shutdown (old bridge may die dirty — ignore errors) then re-ensure.
  try {
    await shutdownBridge(persona);
  } catch {
    // Old bridge may crash on shutdown — that's expected; continue anyway.
  }

  await ensureBridgeRunning(persona);

  // 6. Recheck.
  try {
    const recheckHealth = await fetchHealth(persona);
    if (recheckHealth.version === appVersion) {
      return "restarted";
    }
  } catch {
    // fetchHealth failing after restart is still unresolved.
  }

  return "version_mismatch_unresolved";
}
