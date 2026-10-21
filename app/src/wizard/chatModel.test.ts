import { describe, expect, it } from "vitest";

import { wizardInitModel, wizardModelIsValid } from "./chatModel";

describe("wizard model field", () => {
  it("blank (or spaces) sends no model: the install keeps its current setting", () => {
    expect(wizardModelIsValid("")).toBe(true);
    expect(wizardModelIsValid("   ")).toBe(true);
    expect(wizardInitModel("")).toBeNull();
    expect(wizardInitModel("  ")).toBeNull();
  });

  it.each(["sonnet", "opus", "haiku", "claude-sonnet-5", "claude-3-5-haiku-20241022", " claude-opus-4-8 "])(
    "accepts %j and passes it on trimmed",
    (m) => {
      expect(wizardModelIsValid(m)).toBe(true);
      expect(wizardInitModel(m)).toBe(m.trim());
    },
  );

  it.each(["gpt4", "Claude-Sonnet-5", "claude-", "claude sonnet", "claude-" + "a".repeat(58)])(
    "rejects %j",
    (m) => {
      expect(wizardModelIsValid(m)).toBe(false);
    },
  );
});
