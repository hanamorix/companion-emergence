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

/** The line for a chat model setting in the install's settings file that the app does not support (#446). The
 *  bridge runs the shipped default (Sonnet) instead; picking any model in the picker (the current one included)
 *  replaces the value. */
export function modelUnsupportedMessage(value: string): string {
  return (
    `The model "${value}" in your settings is not supported, so Sonnet is being used instead. ` +
    "Use change to pick a model and replace it."
  );
}

/**
 * The line for a model the Claude CLI rejected while it was in use (from the bridge's
 * `model_rejections`). `chosen` is true when it is the chat model's setting; otherwise it is a model some
 * background job is set to. `jobsOnDefault` is true when the bridge reports that the jobs set to this id now run
 * their shipped default instead (`model_fallbacks`); when it is false the rejected id IS the shipped default of
 * those jobs, so there is nothing to fall back to and they keep failing until Claude Code is updated or the model
 * setting is changed.
 */
export function modelRejectedNowMessage(model: string, chosen: boolean, jobsOnDefault: boolean): string {
  const name = `the model "${shortenModelId(model)}"`;
  if (jobsOnDefault) {
    return (
      `Claude rejected ${name}. It may not exist, or your account may not have access to it. ` +
      (chosen
        ? "Chat now runs the default model instead. Use change to pick another model."
        : "Some background jobs now run the default model instead.")
    );
  }
  return (
    `Claude rejected ${name}, which is the default model. ` +
    (chosen
      ? "Chat keeps failing until Claude Code is updated or you pick another model with change."
      : "Some background jobs keep failing until Claude Code is updated or the model setting is changed.")
  );
}

/** One job that now runs its shipped default because the model it is set to was rejected (from the bridge). */
export interface ModelFallbackInfo {
  job: string;
  configured: string;
  using: string;
}

/** The plain line for a job that fell back to its shipped default model. */
export function modelFallbackMessage(f: ModelFallbackInfo): string {
  return (
    `The ${f.job.replace(/-/g, " ")} job is using "${shortenModelId(f.using)}" instead of ` +
    `"${shortenModelId(f.configured)}", which Claude rejected.`
  );
}
