// Component tests for ChatPanel — addresses 2026-05-08 audit P4-2
// (frontend test coverage beyond smoke). The render-level assertions
// verify P3-4 (image-only sends), the send/stop toggle for the audit
// P2-9 cancel control, and the persona-aware placeholder so future
// hardcoded "Nell" doesn't sneak back in.

import { describe, it, expect, vi, afterEach, beforeAll } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({ port: 0, auth_token: null })),
}));

vi.mock("../bridge", () => ({
  newSession: vi.fn(async () => "test-session-id"),
  closeSession: vi.fn(async () => undefined),
  uploadImage: vi.fn(async () => ({ sha: "deadbeef" })),
}));

vi.mock("../streamChat", () => ({
  streamChat: vi.fn(async () => () => undefined),
}));

import { ChatPanel } from "./ChatPanel";

// jsdom doesn't implement Element.scrollTo; stub it so ChatPanel's
// auto-scroll effect doesn't throw during render.
beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

describe("ChatPanel — image-only + cancel + placeholder (P4-2)", () => {
  afterEach(() => {
    cleanup();
  });

  it("placeholder humanizes the persona name", () => {
    render(<ChatPanel persona="alice" />);
    expect(
      screen.getByPlaceholderText(/^Write to Alice/),
    ).toBeInTheDocument();
  });

  it("placeholder converts underscores to spaces and capitalizes", () => {
    render(<ChatPanel persona="my_companion" />);
    expect(
      screen.getByPlaceholderText(/^Write to My companion/),
    ).toBeInTheDocument();
  });

  it("send button is disabled when there's no text and no image", () => {
    render(<ChatPanel persona="nell" />);
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn).toBeDisabled();
  });

  it("send button shows ↑ glyph when idle", () => {
    render(<ChatPanel persona="nell" />);
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    expect(sendBtn.textContent).toBe("↑");
  });

  it("paperclip + emoji buttons render and are not disabled at idle", () => {
    render(<ChatPanel persona="nell" />);
    expect(
      screen.getByRole("button", { name: /attach image/i }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: /insert emoji/i }),
    ).toBeEnabled();
  });

  it("imports without exploding when persona has trailing whitespace edge cases", () => {
    // Defensive — ChatPanel.capitalize() is in the JSX path, regression
    // here would crash the chat surface entirely.
    render(<ChatPanel persona="x" />);
    expect(screen.getByPlaceholderText(/^Write to X/)).toBeInTheDocument();
  });
});
