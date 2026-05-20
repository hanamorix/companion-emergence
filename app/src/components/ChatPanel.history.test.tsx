// Component tests for ChatPanel's mount-time history hydration —
// v0.0.15-alpha.2 Phase 3B. When the user reopens the app, the bridge
// still has the previous session's JSONL on disk; this surfaces those
// turns back into the messages state so the conversation doesn't appear
// to have evaporated.

import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({ port: 0, auth_token: null })),
}));

vi.mock("../bridge", () => ({
  newSession: vi.fn(async () => "fresh-session"),
  fetchActiveSession: vi.fn(async () => null),
  fetchChatHistory: vi.fn(async () => ({ messages: [], next_before_turn: null })),
  closeSession: vi.fn(async () => undefined),
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

import { ChatPanel } from "./ChatPanel";
import { fetchActiveSession, fetchChatHistory, newSession } from "../bridge";

beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

describe("ChatPanel — mount-time history hydration (Phase 3B)", () => {
  const mockedFetchActive = fetchActiveSession as unknown as ReturnType<typeof vi.fn>;
  const mockedFetchHistory = fetchChatHistory as unknown as ReturnType<typeof vi.fn>;
  const mockedNewSession = newSession as unknown as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockedFetchActive.mockReset();
    mockedFetchHistory.mockReset();
    mockedNewSession.mockReset();
    mockedFetchHistory.mockResolvedValue({ messages: [], next_before_turn: null });
  });

  afterEach(() => {
    cleanup();
  });

  it("calls fetchChatHistory with the active session id when one exists", async () => {
    mockedFetchActive.mockResolvedValue("s_a");
    render(<ChatPanel persona="nell" />);

    await waitFor(() => expect(mockedFetchHistory).toHaveBeenCalled());
    expect(mockedFetchHistory.mock.calls[0]![0]).toBe("nell");
    expect(mockedFetchHistory.mock.calls[0]![1]).toBe("s_a");
    // newSession should NOT have been called — we attached to an existing one.
    expect(mockedNewSession).not.toHaveBeenCalled();
  });

  it("does not call fetchChatHistory when there is no active session", async () => {
    mockedFetchActive.mockResolvedValue(null);
    render(<ChatPanel persona="nell" />);

    // Give effects time to settle; an empty fresh chat should NOT hit
    // history (nothing on disk yet, no need to create a session eagerly).
    await waitFor(() => expect(mockedFetchActive).toHaveBeenCalled());
    expect(mockedFetchHistory).not.toHaveBeenCalled();
    expect(mockedNewSession).not.toHaveBeenCalled();
  });

  it("renders the hydrated messages once history resolves", async () => {
    mockedFetchActive.mockResolvedValue("s_a");
    mockedFetchHistory.mockResolvedValue({
      messages: [
        { role: "user", content: "remembered", turn: 1, ts: "2026-05-20T10:00:00Z" },
        { role: "assistant", content: "yes", turn: 2, ts: "2026-05-20T10:00:05Z" },
      ],
      next_before_turn: null,
    });
    render(<ChatPanel persona="nell" />);

    expect(await screen.findByText("remembered")).toBeInTheDocument();
    expect(await screen.findByText("yes")).toBeInTheDocument();
  });

  it("silently tolerates a history fetch failure (renders empty chat)", async () => {
    mockedFetchActive.mockResolvedValue("s_a");
    mockedFetchHistory.mockRejectedValue(new Error("network flake"));
    render(<ChatPanel persona="nell" />);

    // Settle the rejection so unhandled-rejection warnings don't fire.
    await waitFor(() => expect(mockedFetchHistory).toHaveBeenCalled());
    // The error is non-fatal: the panel still mounts with no messages,
    // matching the fresh-session UX.
    expect(screen.queryByText("remembered")).toBeNull();
  });
});
