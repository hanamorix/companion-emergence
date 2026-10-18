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

/** Shown after Apply when the bridge could not run its check call (timeout, expired login, rate limit...). */
export const MODEL_UNCHECKED_MESSAGE =
  "Couldn't check that model right now; saved anyway, and any problem will show here.";

/**
 * The line for a model the Claude CLI rejected while it was in use (from the bridge's
 * `model_rejections`). `chosen` is true when it is the model the user picked; otherwise it is a model
 * some background job uses on its own, so changing the chat model would not help.
 */
export function modelRejectedNowMessage(model: string, chosen: boolean): string {
  const name = `the model "${shortenModelId(model)}"`;
  return chosen
    ? `Claude rejected ${name}. It may not exist, or your account may not have access to it. Use change to pick another model.`
    : `Claude rejected ${name}, which some background jobs use on their own. Check that your Claude account can use it.`;
}

/** One background tier family whose version-matched model was not available (from the bridge). */
export interface TierFallbackInfo {
  family: string;
  wanted: string;
  using: string;
}

/** The plain line for a tier model that fell back to what that tier uses today. */
export function tierFallbackMessage(f: TierFallbackInfo): string {
  return (
    `Background jobs that use the ${f.family} family are using "${shortenModelId(f.using)}", ` +
    `because "${shortenModelId(f.wanted)}" could not be used ` +
    "(it may not exist, or your account may not have access to it)."
  );
}

/** Derived background-model ids an Apply could not check (its time budget ran out, a timeout, ...). */
export function tiersUncheckedMessage(ids: string[]): string {
  return (
    `Some background models were not checked yet (${ids.map((m) => shortenModelId(m)).join(", ")}). ` +
    "They are checked when first used, and any problem will show here."
  );
}
