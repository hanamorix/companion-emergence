import { describe, it, expect } from "vitest";
import * as streamChatModule from "./streamChat";

describe("streamChat default overall timeout", () => {
  it("is at least 600 seconds (stopgap for long Opus replies)", () => {
    // Read the constant via the test-seam export — added in this phase
    // and retired in Phase 6 when overall_timer becomes idle_timer.
    const value = streamChatModule.__DEFAULT_OVERALL_TIMEOUT_MS__;
    expect(value).toBeGreaterThanOrEqual(600_000);
  });
});
