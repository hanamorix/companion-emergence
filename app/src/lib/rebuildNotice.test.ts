import { describe, expect, it } from "vitest";

import { rebuildNoticeMessage } from "./rebuildNotice";
import { TUNABLES_NOTICE_REASON_MAX } from "./tunablesNotice";

describe("rebuildNoticeMessage", () => {
  it("says what happened, that the saved feelings are in use, and where the log is", () => {
    const text = rebuildNoticeMessage({
      reason: "the first rebuild failed (OSError: disk full)",
      log_path: "/logs/runtime-nell.log",
    });
    expect(text).toBe(
      "Your feelings could not be brought up to date: the first rebuild failed (OSError: disk full). " +
        "The app is using the ones saved before. Details are in the log: /logs/runtime-nell.log.",
    );
  });

  it.each([[null], [""]])("leaves the reason out when it is %j", (reason) => {
    const text = rebuildNoticeMessage({ reason, log_path: "/logs/runtime-nell.log" });
    expect(text).toBe(
      "Your feelings could not be brought up to date. The app is using the ones saved before. " +
        "Details are in the log: /logs/runtime-nell.log.",
    );
  });

  it.each([[null], [""]])("shows the message without a path when log_path is %j", (log_path) => {
    const text = rebuildNoticeMessage({ reason: "the rebuilt state could not be written", log_path });
    expect(text).toBe(
      "Your feelings could not be brought up to date: the rebuilt state could not be written. " +
        "The app is using the ones saved before.",
    );
    expect(text).not.toContain("Details are in the log");
  });

  it("removes control characters from the reason and cuts it, but keeps a long path whole", () => {
    const longPath = "/logs/" + "d".repeat(300) + "/runtime-nell.log";
    const text = rebuildNoticeMessage({
      reason: "a\u0000b\u0007c" + "x".repeat(300),
      log_path: longPath,
    });
    expect(text).not.toMatch(/[\u0000-\u001f]/);
    expect(text).toContain("abc" + "x".repeat(TUNABLES_NOTICE_REASON_MAX - 3));
    expect(text).not.toContain("x".repeat(TUNABLES_NOTICE_REASON_MAX + 1));
    expect(text).toContain(`Details are in the log: ${longPath}.`);
  });

  it("uses no em-dash", () => {
    for (const notice of [
      { reason: "x", log_path: "/p" },
      { reason: null, log_path: null },
    ]) {
      expect(rebuildNoticeMessage(notice)).not.toContain("—");
    }
  });
});
