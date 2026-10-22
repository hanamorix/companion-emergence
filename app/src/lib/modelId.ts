/**
 * The chat model setting: one of three words (sonnet / opus / haiku, the picker's vocabulary) or a full model id
 * (it starts with `claude-`), stored in the install's settings file (tunables.toml) and passed to `claude --model` as written.
 * The app holds no model id of its own: the brain resolves the words.
 *
 * KEEP IN SYNC with brain/bridge/model_tier.py (`KNOWN_MODELS`, `MODEL_ID_PATTERN`, `MODEL_ID_MAX_LENGTH`). The pattern is
 * anchored with `^...$`; without the `m` flag a JS `$` matches only at the very end, so a trailing
 * newline is rejected, exactly like Python's `fullmatch`.
 */
export const CHAT_MODEL_ALIASES = ["sonnet", "opus", "haiku"] as const;
export const CHAT_MODEL_ID_PATTERN = /^claude-[a-z0-9.-]+$/;

/** Longest full id accepted (brain/bridge/model_tier.py `MODEL_ID_MAX_LENGTH`, enforced on every write path).
 *  Real ids are well under it. */
export const MODEL_ID_MAX_LENGTH = 64;

export function isChatModelAlias(value: string): boolean {
  return (CHAT_MODEL_ALIASES as readonly string[]).includes(value);
}

/** The words a save runs NO check call for: `sonnet` removes the override (the chat job's shipped default) and `haiku`
 *  is written as the alias. `opus` is written as its full id and every full id is written as typed, so both are
 *  checked first (brain/server.py `_apply_chat_model`). KEEP IN SYNC with that rule. */
export const CHECK_FREE_ALIASES = ["sonnet", "haiku"] as const;

export function isCheckFreeAlias(value: string): boolean {
  return (CHECK_FREE_ALIASES as readonly string[]).includes(value);
}

export function isValidChatModel(value: string): boolean {
  return (
    isChatModelAlias(value) ||
    (value.length <= MODEL_ID_MAX_LENGTH && CHAT_MODEL_ID_PATTERN.test(value))
  );
}

/** Display form of an id for an error line: very long ids are cut with an ellipsis. */
export function shortenModelId(id: string, max: number = MODEL_ID_MAX_LENGTH): string {
  return id.length > max ? `${id.slice(0, max)}...` : id;
}
