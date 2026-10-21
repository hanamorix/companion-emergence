import { MODEL_ID_MAX_LENGTH, isValidChatModel } from "../lib/modelId";

/**
 * The first-run wizard's optional "Claude model" field. Blank means keep the install's current setting;
 * anything else must be a word (sonnet, opus, haiku) or a full model id (same rule as the picker). It is not
 * checked against Claude here: `nell init` only validates the syntax, and a full id is checked when it is
 * first used.
 */
export function wizardModelIsValid(input: string): boolean {
  const v = input.trim();
  return v === "" || isValidChatModel(v);
}

/** The value to hand to `nell init --model`, or null (no flag) when the field is blank. */
export function wizardInitModel(input: string): string | null {
  const v = input.trim();
  return v === "" ? null : v;
}

export { MODEL_ID_MAX_LENGTH };
