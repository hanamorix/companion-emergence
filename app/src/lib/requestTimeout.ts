/** The time limit on one short request to the bridge (a /health probe, the stop request): 3 s (#456 review M1). A
 *  wedged bridge can accept the connection and never answer. Its own module so code that mocks `bridge.ts` in tests
 *  still sees the value. */
export const REQUEST_TIMEOUT_MS = 3000;
