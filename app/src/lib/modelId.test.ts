import { describe, expect, it } from "vitest";

import {
  CHAT_MODEL_ID_PATTERN,
  MODEL_ID_MAX_LENGTH,
  isChatModelAlias,
  isValidChatModel,
  shortenModelId,
} from "./modelId";

// KEEP IN SYNC with FULL_IDS / BAD_IDS in tests/bridge/test_persona_config_model.py
// (the bridge validates with the same grammar).
const FULL_IDS = ["claude-sonnet-5", "claude-opus-4-8", "claude-3-5-haiku-20241022", "claude-sonnet-4.5"];
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

  it("does not accept a trailing newline (no multiline flag)", () => {
    expect(CHAT_MODEL_ID_PATTERN.test("claude-sonnet-5\n")).toBe(false);
  });

  it("shortens only ids longer than the display limit", () => {
    const long = "claude-" + "x".repeat(100);
    expect(shortenModelId("claude-sonnet-5")).toBe("claude-sonnet-5");
    expect(shortenModelId(long)).toBe(long.slice(0, MODEL_ID_MAX_LENGTH) + "...");
  });
});
