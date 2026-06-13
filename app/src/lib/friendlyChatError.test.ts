import { describe, expect, it } from "vitest";

import { friendlyChatError } from "./friendlyChatError";

describe("friendlyChatError", () => {
  it("maps provider_failed to actionable Claude guidance", () => {
    const out = friendlyChatError("provider_failed");
    expect(out).toMatch(/Claude Code is installed/i);
    expect(out).toMatch(/signed in/i);
    expect(out).toMatch(/restart the bridge/i);
  });

  it("passes other errors through unchanged", () => {
    expect(friendlyChatError("Bridge unreachable: Failed to fetch")).toBe(
      "Bridge unreachable: Failed to fetch",
    );
  });
});
