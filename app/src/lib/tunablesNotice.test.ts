import { describe, expect, it } from "vitest";

import { TUNABLES_NOTICE_REASON_MAX, tunablesNoticeMessage } from "./tunablesNotice";

describe("tunablesNoticeMessage", () => {
  it("names the file, the line and column and the reason, and says the file was left as it is", () => {
    const text = tunablesNoticeMessage({ kind: "unreadable", line: 7, column: 12, reason: "Invalid statement" });
    expect(text).toContain("tunables.toml");
    expect(text).toContain("at line 7, column 12");
    expect(text).toContain("(Invalid statement)");
    expect(text).toContain("your settings are not in effect and the defaults are being used");
    expect(text).toContain("The file was left as it is");
    expect(text).toContain("Fix that line to use your settings again");
  });

  it("names the line alone when there is no column", () => {
    const text = tunablesNoticeMessage({ kind: "unreadable", line: 3, column: null, reason: "Duplicate key" });
    expect(text).toContain("at line 3");
    expect(text).not.toContain("column");
  });

  it("says to fix the file when the parser gave no position", () => {
    const text = tunablesNoticeMessage({ kind: "unreadable", line: null, column: null, reason: null });
    expect(text).toContain("tunables.toml could not be read,");
    expect(text).not.toContain("at line");
    expect(text).not.toContain("()");
    expect(text).toContain("Fix the file to use your settings again");
  });

  it("uses no em-dash and does not mention the old file name or a kept-aside copy", () => {
    for (const notice of [
      { kind: "unreadable" as const, line: 1, column: 2, reason: "x" },
      { kind: "unreadable" as const, line: null, column: null, reason: null },
    ]) {
      const text = tunablesNoticeMessage(notice);
      expect(text).not.toContain("—");
      expect(text).not.toContain("tunables.json");
      expect(text).not.toContain("kept as");
    }
  });

  it("removes control characters from the reason and cuts it", () => {
    const text = tunablesNoticeMessage({
      kind: "unreadable",
      line: 1,
      column: 1,
      reason: "a\u0000b\u0007c" + "x".repeat(300),
    });
    expect(text).not.toMatch(/[\u0000-\u001f]/);
    expect(text).toContain("abc" + "x".repeat(TUNABLES_NOTICE_REASON_MAX - 3));
    expect(text).not.toContain("x".repeat(TUNABLES_NOTICE_REASON_MAX + 1));
  });
});
