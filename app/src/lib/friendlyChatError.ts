/**
 * Map raw chat error strings to actionable user-facing guidance.
 *
 * The bridge strips provider error detail server-side (audit 2026-05-07
 * P3-2), so a failed `claude` subprocess surfaces as the bare, unactionable
 * code `provider_failed`. Turn the known codes into something the user can
 * act on; pass everything else through unchanged.
 */
export function friendlyChatError(raw: string): string {
  if (/auth_expired/i.test(raw)) {
    // #246: the brain's own Claude login lapsed — a re-login, not a reinstall.
    return (
      "Your companion's Claude login has expired. Re-authorise from the " +
      "connection panel (the bridge restarts itself afterwards)."
    );
  }
  if (/provider_failed/i.test(raw)) {
    return (
      "Your companion couldn't reach Claude. Make sure Claude Code is installed " +
      "and you're signed in (run `claude` in a terminal once to sign in), then " +
      "restart the bridge from the connection panel."
    );
  }
  return raw;
}
