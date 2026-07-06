// Phase 4 — v0.0.15-alpha.2 chat-reliability.
//
// A Linux user reported "(Nell couldn't answer — see the error below.)"
// rendering with no error text beneath it. The root cause was that
// streamChat's onError sometimes fires with an empty string (or a
// stringified Error with no message), and setError("") leaves the
// error banner blank. setErrorSafe wraps setError to map empty /
// whitespace-only strings to a fixed copy pointing at the bridge
// restart button.

import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({ port: 0, auth_token: null })),
}));

vi.mock("../bridge", () => ({
  newSession: vi.fn(async () => "test-session-id"),
  fetchActiveSession: vi.fn(async () => null),
  fetchChatHistory: vi.fn(async () => ({ messages: [], next_before_turn: null })),
  snapshotSession: vi.fn(async () => ({ closed: false, errors: 0 })),
  uploadImage: vi.fn(async () => ({ sha: "deadbeef" })),
  getBridgeCredentials: vi.fn(async () => ({
    url: "http://127.0.0.1:50000",
    port: 50000,
    authToken: "test-token",
  })),
}));

vi.mock("../bridgeEvents", () => ({
  subscribeToBridgeEvents: vi.fn(() => ({
    subscribe: () => () => undefined,
    close: () => undefined,
  })),
}));

vi.mock("../streamChat", () => ({
  streamChat: vi.fn(async () => () => undefined),
}));

// ChatPanel's glass header (Phase 4) resolves an avatar thumb via
// expressions.ts, whose import.meta.glob eagerly loads every expression
// PNG at module-load time. This worktree checkout's .git is a file (not
// a directory), which breaks Vite's fs.allow root-detection and denies
// that glob's fetch outside the app/ dir. Stub the module — these tests
// exercise chat behavior, not expression art.
vi.mock("../expressions", () => ({
  resolveFrameUrl: () => "",
}));

import { ChatPanel } from "./ChatPanel";
import { streamChat } from "../streamChat";

beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

describe("ChatPanel defensive empty-error copy (Phase 4)", () => {
  const mockedStreamChat = streamChat as unknown as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockedStreamChat.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  async function typeAndSend(persona: string) {
    render(<ChatPanel persona={persona} />);
    const textarea = screen.getByPlaceholderText(
      /^Write to/,
    ) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: "hello" } });
    });
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await act(async () => {
      fireEvent.click(sendBtn);
    });
    await waitFor(() => {
      expect(mockedStreamChat).toHaveBeenCalled();
    });
  }

  it("renders fallback copy when streamChat fires onError('')", async () => {
    mockedStreamChat.mockImplementation(
      async (
        _p: string,
        _s: string,
        _m: string,
        handlers: { onError?: (msg: string) => void },
      ) => {
        setTimeout(() => handlers.onError?.(""), 0);
        return () => undefined;
      },
    );

    await typeAndSend("nell");

    await waitFor(() => {
      expect(
        screen.getByText(/bridge couldn't respond|supervisor may have stalled/i),
      ).toBeInTheDocument();
    });
  });

  it("renders fallback copy when streamChat fires onError with whitespace only", async () => {
    mockedStreamChat.mockImplementation(
      async (
        _p: string,
        _s: string,
        _m: string,
        handlers: { onError?: (msg: string) => void },
      ) => {
        setTimeout(() => handlers.onError?.("   \n\t  "), 0);
        return () => undefined;
      },
    );

    await typeAndSend("nell");

    await waitFor(() => {
      expect(
        screen.getByText(/bridge couldn't respond|supervisor may have stalled/i),
      ).toBeInTheDocument();
    });
  });

  it("renders the original error when non-empty", async () => {
    mockedStreamChat.mockImplementation(
      async (
        _p: string,
        _s: string,
        _m: string,
        handlers: { onError?: (msg: string) => void },
      ) => {
        setTimeout(() => handlers.onError?.("provider exploded"), 0);
        return () => undefined;
      },
    );

    await typeAndSend("nell");

    await waitFor(() => {
      expect(screen.getByText(/provider exploded/i)).toBeInTheDocument();
    });
  });
});
