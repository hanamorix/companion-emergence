import { describe, expect, it } from "vitest";

import { rebuildNoticeMessage } from "./rebuildNotice";
import { TUNABLES_NOTICE_REASON_MAX } from "./tunablesNotice";

describe("rebuildNoticeMessage", () => {
  it("says what happened, that the saved feelings are in use, and where the log is", () => {
    const text = rebuildNoticeMessage(
      {
        reason: "the first rebuild failed (OSError: disk full)",
        log_path: "/logs/runtime-nell.log",
      },
      "Nell",
    );
    expect(text).toBe(
      "Nell's feelings could not be brought up to date: the first rebuild failed (OSError: disk full). " +
        "The app is using the ones saved before. Details are in the log: /logs/runtime-nell.log.",
    );
  });

  it.each([[null], [""]])("leaves the reason out when it is %j", (reason) => {
    const text = rebuildNoticeMessage({ reason, log_path: "/logs/runtime-nell.log" }, "Nell");
    expect(text).toBe(
      "Nell's feelings could not be brought up to date. The app is using the ones saved before. " +
        "Details are in the log: /logs/runtime-nell.log.",
    );
  });

  it.each([[null], [""]])("shows the message without a path when log_path is %j", (log_path) => {
    const text = rebuildNoticeMessage({ reason: "the rebuilt state could not be written", log_path }, "Nell");
    expect(text).toBe(
      "Nell's feelings could not be brought up to date: the rebuilt state could not be written. " +
        "The app is using the ones saved before.",
    );
    expect(text).not.toContain("Details are in the log");
  });

  it("names the persona it is given, as given", () => {
    const text = rebuildNoticeMessage({ reason: null, log_path: null }, "Mira");
    expect(text).toBe("Mira's feelings could not be brought up to date. The app is using the ones saved before.");
  });

  it.each([[""], ["   "]])("falls back to Your when the name is %j", (name) => {
    const text = rebuildNoticeMessage({ reason: null, log_path: null }, name);
    expect(text).toBe("Your feelings could not be brought up to date. The app is using the ones saved before.");
    expect(text).not.toMatch(/^'s/);
  });

  it("removes control characters from the reason and cuts it, but keeps a long path whole", () => {
    const longPath = "/logs/" + "d".repeat(300) + "/runtime-nell.log";
    const text = rebuildNoticeMessage(
      {
        reason: "a\u0000b\u0007c" + "x".repeat(300),
        log_path: longPath,
      },
      "Nell",
    );
    expect(text).not.toMatch(/[\u0000-\u001f]/);
    expect(text).toContain("abc" + "x".repeat(TUNABLES_NOTICE_REASON_MAX - 3));
    expect(text).not.toContain("x".repeat(TUNABLES_NOTICE_REASON_MAX + 1));
    expect(text).toContain(`Details are in the log: ${longPath}.`);
  });

  it("uses no em-dash", () => {
    for (const name of ["Nell", ""]) {
      for (const notice of [
        { reason: "x", log_path: "/p" },
        { reason: null, log_path: null },
      ]) {
        expect(rebuildNoticeMessage(notice, name)).not.toContain("—");
      }
    }
  });
});
