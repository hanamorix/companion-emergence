// #532: a long reach-out or voice-edit proposal grew its card past the chat panel's fixed height, and the panel clipped
// the card's buttons, so the user could not answer. The cards now sit in one stack that can shrink and scroll, below the
// messages and above the input row. jsdom does no layout, so these tests pin the structure and the stack's inline styles;
// the CSS caps inside each card are pinned in chatCards.css.test.ts.

import { describe, it, expect, vi, afterEach, beforeAll } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({ port: 0, auth_token: null })),
}));

vi.mock("../bridge", () => ({
  newSession: vi.fn(async () => "test-session-id"),
  fetchActiveSession: vi.fn(async () => null),
  fetchChatHistory: vi.fn(async () => ({ messages: [], next_before_turn: null })),
  closeSession: vi.fn(async () => undefined),
  snapshotSession: vi.fn(async () => ({ closed: false, errors: 0 })),
  getBridgeCredentials: vi.fn(async () => ({ url: "http://127.0.0.1:50000", port: 50000, authToken: "t" })),
  acceptVoiceEdit: vi.fn(async () => ({ ok: true })),
  rejectVoiceEdit: vi.fn(async () => ({ ok: true })),
}));

vi.mock("../bridgeEvents", () => ({
  subscribeToBridgeEvents: vi.fn(() => ({ subscribe: () => () => undefined, close: () => undefined })),
}));

// The header's avatar resolves expression art through import.meta.glob; these tests are about layout, not art.
vi.mock("../expressions", () => ({ resolveFrameUrl: () => "" }));

import { ChatPanel } from "./ChatPanel";

beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

afterEach(() => {
  cleanup();
});

function makeStream() {
  const handlers = new Set<(e: Record<string, unknown> & { type: string }) => void>();
  return {
    subscribe(h: (e: Record<string, unknown> & { type: string }) => void) {
      handlers.add(h);
      return () => handlers.delete(h);
    },
    emit(e: Record<string, unknown> & { type: string }) {
      for (const h of handlers) h(e);
    },
  };
}

const LONG = "a long thought that keeps going. ".repeat(80);

async function renderWithBothCards() {
  const stream = makeStream();
  const view = render(<ChatPanel persona="nell" eventStream={stream} />);
  await act(async () => {
    stream.emit({
      type: "initiate_delivered",
      audit_id: "ia_long",
      body: LONG,
      urgency: "quiet",
      state: "delivered",
      timestamp: "2026-11-24T14:32:00+00:00",
    });
    stream.emit({
      type: "initiate_delivered",
      kind: "voice_edit_proposal",
      audit_id: "ia_ve_long",
      body: LONG,
      diff: "- I'm fine when tired\n+ I get quieter when tired",
    });
  });
  return view;
}

describe("ChatPanel — card stack (#532)", () => {
  it("puts the reach-out and the voice-edit cards in one stack", async () => {
    await renderWithBothCards();
    const stack = screen.getByTestId("chat-cards");
    expect(stack).toContainElement(screen.getByRole("region", { name: /nell reached out/i }));
    expect(stack).toContainElement(screen.getByRole("dialog", { name: /voice edit proposal/i }));
  });

  it("lets the stack shrink and scroll instead of pushing the input row out", async () => {
    await renderWithBothCards();
    const stack = screen.getByTestId("chat-cards") as HTMLElement;
    expect(stack.style.flex).toBe("0 1 auto");
    expect(stack.style.minHeight).toMatch(/^0(px)?$/);
    expect(stack.style.maxHeight).toBeTruthy();
    expect(stack.style.overflowY).toBe("auto");
  });
});
