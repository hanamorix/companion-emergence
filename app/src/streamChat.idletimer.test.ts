import { describe, it, expect } from "vitest";
import * as streamChatModule from "./streamChat";

describe("streamChat idle-timer", () => {
  it("DEFAULT_IDLE_TIMEOUT_MS is 60_000", () => {
    expect(streamChatModule.__DEFAULT_IDLE_TIMEOUT_MS__).toBe(60_000);
  });

  it("__DEFAULT_OVERALL_TIMEOUT_MS__ is no longer exported (replaced by idle timer)", () => {
    expect((streamChatModule as Record<string, unknown>).__DEFAULT_OVERALL_TIMEOUT_MS__).toBeUndefined();
  });
});
