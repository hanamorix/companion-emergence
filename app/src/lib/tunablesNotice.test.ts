import { describe, expect, it } from "vitest";

import { TUNABLES_NOTICE_NAME_MAX, tunablesNoticeMessage } from "./tunablesNotice";

const NAME = "tunables.json.corrupt-20261019T120000Z";

describe("tunablesNoticeMessage", () => {
  it("names the kept file and says how to dismiss a reset", () => {
    const text = tunablesNoticeMessage({ kind: "reset", kept_as: NAME });
    expect(text).toContain(NAME);
    expect(text).toContain("your overrides are not in effect");
    expect(text).toContain("delete that file to dismiss this message");
  });

  it("names the kept file and says how to dismiss a restore", () => {
    const text = tunablesNoticeMessage({ kind: "restored", kept_as: NAME });
    expect(text).toContain(NAME);
    expect(text).toContain("older saved copy");
    expect(text).toContain("delete that file to dismiss this message");
  });

  it("says an unreadable file was left as it is and goes away after the restart", () => {
    const text = tunablesNoticeMessage({ kind: "unreadable", kept_as: null });
    expect(text).toContain("left as it is");
    expect(text).toContain("after the restart");
  });

  it("uses no em-dash in any text", () => {
    for (const kind of ["reset", "restored", "unreadable"] as const) {
      expect(tunablesNoticeMessage({ kind, kept_as: NAME })).not.toContain("—");
    }
  });

  it("removes control characters from the name and cuts it", () => {
    const text = tunablesNoticeMessage({ kind: "reset", kept_as: "a\u0000b\u0007c" + "x".repeat(200) });
    expect(text).not.toMatch(/[\u0000-\u001f]/);
    expect(text).toContain("abc" + "x".repeat(TUNABLES_NOTICE_NAME_MAX - 3));
    expect(text).not.toContain("x".repeat(TUNABLES_NOTICE_NAME_MAX + 1));
  });
});
