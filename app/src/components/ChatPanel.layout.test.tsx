import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { ChatPanel } from "./ChatPanel";
import * as bridge from "../bridge";

vi.mock("../streamChat", () => ({
  streamChat: vi.fn(async () => () => undefined),
}));

beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(bridge, "getBridgeCredentials").mockResolvedValue({
    port: 1234,
    authToken: "t",
  } as any);
  vi.spyOn(bridge, "newSession").mockResolvedValue("s_a" as any);
  vi.spyOn(bridge, "fetchChatHistory").mockResolvedValue({
    messages: [],
    next_before_turn: null,
  });
});

describe("ChatPanel layout", () => {
  it("messages container has data-testid='chat-messages'", () => {
    const { container } = render(<ChatPanel persona="nell" />);
    const list = container.querySelector('[data-testid="chat-messages"]');
    expect(list).not.toBeNull();
  });

  it("messages container uses flex 1 1 auto with min-height 0", () => {
    const { container } = render(<ChatPanel persona="nell" />);
    const list = container.querySelector('[data-testid="chat-messages"]') as HTMLElement;
    expect(list.style.flex).toBe("1 1 auto");
    expect(list.style.minHeight).toMatch(/^0(px)?$/);
  });
});
