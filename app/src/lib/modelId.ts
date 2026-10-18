/**
 * The chat model setting: an alias (sonnet / opus / haiku) or a full model id such as
 * `claude-sonnet-5`, stored and passed to `claude --model` as written.
 *
 * KEEP IN SYNC with brain/persona_config.py (`KNOWN_MODELS`, `MODEL_ID_PATTERN`). The pattern is
 * anchored with `^...$`; without the `m` flag a JS `$` matches only at the very end, so a trailing
 * newline is rejected, exactly like Python's `fullmatch`.
 */
export const CHAT_MODEL_ALIASES = ["sonnet", "opus", "haiku"] as const;
export const CHAT_MODEL_ID_PATTERN = /^claude-[a-z0-9.-]+$/;

/** Longest id the picker accepts typing and the error text shows in full. Real ids are well under it. */
export const MODEL_ID_MAX_LENGTH = 64;

export function isChatModelAlias(value: string): boolean {
  return (CHAT_MODEL_ALIASES as readonly string[]).includes(value);
}

export function isValidChatModel(value: string): boolean {
  return isChatModelAlias(value) || CHAT_MODEL_ID_PATTERN.test(value);
}

/** Display form of an id for an error line: very long ids are cut with an ellipsis. */
export function shortenModelId(id: string, max: number = MODEL_ID_MAX_LENGTH): string {
  return id.length > max ? `${id.slice(0, max)}...` : id;
}
