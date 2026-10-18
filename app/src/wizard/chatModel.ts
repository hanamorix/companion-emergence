import { MODEL_ID_MAX_LENGTH, isValidChatModel } from "../lib/modelId";

/**
 * The first-run wizard's optional "Claude model" field. Blank means the default (sonnet); anything else
 * must be an alias or a full model id (same rule as the picker). It is not checked against Claude here:
 * `nell init` only validates the syntax, and a full id is checked when it is first used.
 */
export const DEFAULT_WIZARD_MODEL = "sonnet";

export function wizardModelIsValid(input: string): boolean {
  const v = input.trim();
  return v === "" || isValidChatModel(v);
}

/** The value to hand to `nell init --model`. */
export function wizardInitModel(input: string, fallback: string = DEFAULT_WIZARD_MODEL): string {
  const v = input.trim();
  return v === "" ? fallback : v;
}

export { MODEL_ID_MAX_LENGTH };
