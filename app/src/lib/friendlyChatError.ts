import { shortenModelId } from "./modelId";

/**
 * Where the "model not recognised" message appears decides what it tells the user to do next.
 * "chat": the model was already saved and a chat turn failed with it. "picker": the user just
 * applied it and it was refused, so nothing was saved.
 */
export type ModelRejectionWhere = "chat" | "picker";

/**
 * The specific message for a model the Claude CLI rejected (unknown id, or not available to the
 * signed-in account; the CLI does not say which). Names the model and what to do next.
 */
export function modelRejectionMessage(
  model: string | null | undefined,
  where: ModelRejectionWhere,
): string {
  const subject = model ? `the model "${shortenModelId(model)}"` : "the chosen model";
  const next =
    where === "chat"
      ? "Check the model ID in the Model section of the connection panel, or switch back to Sonnet, Opus or Haiku."
      : "Check the model ID, or pick Sonnet, Opus or Haiku.";
  return `Claude doesn't recognise ${subject}, or your account can't use it. ${next}`;
}

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
  // #440: `model_unavailable` or `model_unavailable:<model id>`. The id alphabet matches the
  // bridge's model-id grammar, so only a valid id can ever be echoed here.
  const rejected = /model_unavailable(?::([a-z0-9.-]+))?/i.exec(raw);
  if (rejected) {
    return modelRejectionMessage(rejected[1], "chat");
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
