/** The time limit on one short request to the bridge (a /health probe, the stop request): 3 s (#456 review M1). A
 *  wedged bridge can accept the connection and never answer. Its own module so code that mocks `bridge.ts` in tests
 *  still sees the value. */
export const REQUEST_TIMEOUT_MS = 3000;

/** How long the app waits for a starting bridge's /health, and how often it asks (the restart flow and a start that
 *  found another start already under way, #456 review R2-M1). The wait's budget is paused while the engine's start-up
 *  record shows in the bridge status (lib/engineProgress.ts), so this is the wait once nothing else is running. */
export const HEALTH_WAIT_MS = 30000;
export const HEALTH_POLL_INTERVAL_MS = 500;

/** How often the app reads `nell supervisor status --json` while it follows a bridge (a stop, a start). Mirrors
 *  `START_STATUS_POLL_MS` in src-tauri/src/lib.rs (the Rust start wait reads the same status at the same pace). */
export const STATUS_POLL_MS = 5000;
