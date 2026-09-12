import { describe, expect, it } from "vitest";

import { friendlyChatError } from "./friendlyChatError";

describe("friendlyChatError", () => {
  it("maps provider_failed to actionable Claude guidance", () => {
    const out = friendlyChatError("provider_failed");
    expect(out).toMatch(/Claude Code is installed/i);
    expect(out).toMatch(/signed in/i);
    expect(out).toMatch(/restart the bridge/i);
  });

  it("maps auth_expired to a re-authorise instruction (#246)", () => {
    const out = friendlyChatError("auth_expired");
    expect(out).toMatch(/login/i);
    expect(out).toMatch(/re-?authori[sz]e/i);
    expect(out).toMatch(/connection panel/i);
  });

  it("passes other errors through unchanged", () => {
    expect(friendlyChatError("Bridge unreachable: Failed to fetch")).toBe(
      "Bridge unreachable: Failed to fetch",
    );
  });
});
