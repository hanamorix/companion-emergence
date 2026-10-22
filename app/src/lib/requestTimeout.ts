/** The time limit on one short request to the bridge (a /health probe, the stop request): 3 s (#456 review M1). A
 *  wedged bridge can accept the connection and never answer. Its own module so code that mocks `bridge.ts` in tests
 *  still sees the value. */
export const REQUEST_TIMEOUT_MS = 3000;

/** How long the app waits for a starting bridge's /health, and how often it asks (the restart flow and a start that
 *  found another start already under way, #456 review R2-M1). */
export const HEALTH_WAIT_MS = 30000;
export const HEALTH_POLL_INTERVAL_MS = 500;
