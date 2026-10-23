import { describe, expect, it } from "vitest";

import {
  CHAT_MODEL_ID_PATTERN,
  MODEL_ID_MAX_LENGTH,
  CHECK_FREE_ALIASES,
  isChatModelAlias,
  isCheckFreeAlias,
  isValidChatModel,
  shortenModelId,
} from "./modelId";

// KEEP IN SYNC with FULL_IDS / BAD_IDS in tests/unit/brain/bridge/test_model_tier.py (a Python test reads these two
// arrays out of this file and fails when they differ; the bridge validates with the same grammar and length cap).
const FULL_IDS = [
  "claude-sonnet-5",
  "claude-cli",
  "claude-opus-4-8",
  "claude-3-5-haiku-20241022",
  "claude-sonnet-4.5",
  "claude-" + "a".repeat(57), // exactly 64 characters: the longest accepted id
];
const BAD_IDS = [
  "gpt4",
  "Claude-Sonnet-5",
  "claude-",
  "claude-sonnet-5\n",
  " claude-sonnet-5",
  "claude-sonnet-5 ",
  "claude_sonnet_5",
  "claude-sonnet 5",
  "claude-sonnet-5;ls",
  "claude-Sonnet",
  "sonnet[1m]",
  "opusplan",
  "claude-sonnet-5[1m]",
  "claude-" + "a".repeat(58), // 65 characters: one over the cap
  "claude-" + "a".repeat(5000),
  "",
];

describe("modelId", () => {
  it.each(["sonnet", "opus", "haiku"])("accepts the alias %s", (alias) => {
    expect(isChatModelAlias(alias)).toBe(true);
    expect(isValidChatModel(alias)).toBe(true);
  });

  it.each(FULL_IDS)("accepts the full id %s", (id) => {
    expect(isValidChatModel(id)).toBe(true);
    expect(isChatModelAlias(id)).toBe(false);
  });

  it.each(BAD_IDS)("rejects %j", (bad) => {
    expect(isValidChatModel(bad)).toBe(false);
  });

  it("names the two words that never run a check call: sonnet and haiku, not opus and not a full id", () => {
    expect([...CHECK_FREE_ALIASES]).toEqual(["sonnet", "haiku"]);
    expect(isCheckFreeAlias("sonnet")).toBe(true);
    expect(isCheckFreeAlias("haiku")).toBe(true);
    expect(isCheckFreeAlias("opus")).toBe(false);
    expect(isCheckFreeAlias("claude-sonnet-5")).toBe(false);
  });

  it("caps a full id at 64 characters", () => {
    expect(MODEL_ID_MAX_LENGTH).toBe(64);
    expect("claude-" + "a".repeat(57)).toHaveLength(64);
  });

  it("does not accept a trailing newline (no multiline flag)", () => {
    expect(CHAT_MODEL_ID_PATTERN.test("claude-sonnet-5\n")).toBe(false);
  });

  it("shortens only ids longer than the display limit", () => {
    const long = "claude-" + "x".repeat(100);
    expect(shortenModelId("claude-sonnet-5")).toBe("claude-sonnet-5");
    expect(shortenModelId(long)).toBe(long.slice(0, MODEL_ID_MAX_LENGTH) + "...");
  });
});
