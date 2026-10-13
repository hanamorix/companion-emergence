// #301 — a failed turn must not lose what the user wrote. The composer is
// cleared the moment Send is pressed; when the turn then fails (the stream
// reports an error, the call throws, or no session can be created) the
// message text — and any attached files — go back into the composer, unless
// the user has already started writing something else.

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
  uploadImage: vi.fn(async () => ({ sha: "deadbeef", kind: "image", media_type: "image/png" })),
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

vi.mock("../expressions", () => ({
  resolveFrameUrl: () => "",
}));

import { ChatPanel } from "./ChatPanel";
import { streamChat } from "../streamChat";
import { newSession } from "../bridge";

type Handlers = { onChunk?: (t: string) => void; onDone?: () => void; onError?: (msg: string) => void };

beforeAll(() => {
  if (!Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function () {} as Element["scrollTo"];
  }
});

describe("ChatPanel — a failed turn gives the message back (#301)", () => {
  const mockedStreamChat = streamChat as unknown as ReturnType<typeof vi.fn>;
  const mockedNewSession = newSession as unknown as ReturnType<typeof vi.fn>;
  let urlCounter = 0;

  beforeEach(() => {
    mockedStreamChat.mockReset();
    mockedNewSession.mockReset();
    mockedNewSession.mockImplementation(async () => "test-session-id");
    urlCounter = 0;
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => `blob:test-${++urlCounter}`);
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  function textarea() {
    return screen.getByPlaceholderText(/^Write to/) as HTMLTextAreaElement;
  }

  async function typeAndSend(text = "hello there") {
    await act(async () => {
      fireEvent.change(textarea(), { target: { value: text } });
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^send$/i }));
    });
  }

  function failStreamWith(msg: string) {
    mockedStreamChat.mockImplementation(async (_p: string, _s: string, _m: string, h: Handlers) => {
      setTimeout(() => h.onError?.(msg), 0);
      return () => undefined;
    });
  }

  it("puts the text back when the stream reports an error", async () => {
    failStreamWith("provider exploded");
    render(<ChatPanel persona="nell" />);
    await typeAndSend();
    await waitFor(() => expect(screen.getByText(/provider exploded/i)).toBeInTheDocument());
    expect(textarea().value).toBe("hello there");
  });

  it("puts the text back when the send throws", async () => {
    mockedStreamChat.mockImplementation(async () => {
      throw new Error("socket closed");
    });
    render(<ChatPanel persona="nell" />);
    await typeAndSend();
    await waitFor(() => expect(screen.getByText(/socket closed/i)).toBeInTheDocument());
    expect(textarea().value).toBe("hello there");
  });

  it("puts the text back when no session can be created", async () => {
    mockedNewSession.mockImplementation(async () => {
      throw new Error("connection refused");
    });
    render(<ChatPanel persona="nell" />);
    await typeAndSend();
    await waitFor(() => expect(screen.getByText(/connection refused/i)).toBeInTheDocument());
    expect(textarea().value).toBe("hello there");
  });

  it("never overwrites something the user started writing after sending", async () => {
    let fail: (msg: string) => void = () => undefined;
    mockedStreamChat.mockImplementation(async (_p: string, _s: string, _m: string, h: Handlers) => {
      fail = (msg) => h.onError?.(msg);
      return () => undefined;
    });
    render(<ChatPanel persona="nell" />);
    await typeAndSend();
    await act(async () => {
      fireEvent.change(textarea(), { target: { value: "a new thought" } });
    });
    await act(async () => fail("provider exploded"));
    expect(textarea().value).toBe("a new thought");
  });

  it("leaves the composer empty after a successful turn", async () => {
    mockedStreamChat.mockImplementation(async (_p: string, _s: string, _m: string, h: Handlers) => {
      setTimeout(() => {
        h.onChunk?.("hi!");
        h.onDone?.();
      }, 0);
      return () => undefined;
    });
    render(<ChatPanel persona="nell" />);
    await typeAndSend();
    await waitFor(() => expect(screen.getByText("hi!")).toBeInTheDocument());
    expect(textarea().value).toBe("");
  });

  it("puts an attached image back too", async () => {
    failStreamWith("provider exploded");
    const { container } = render(<ChatPanel persona="nell" />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File(["x"], "shot.png", { type: "image/png" })] } });
    });
    await waitFor(() => expect(screen.getByAltText("shot.png")).toBeInTheDocument());
    await typeAndSend("look at this");
    await waitFor(() => expect(screen.getByText(/provider exploded/i)).toBeInTheDocument());
    expect(screen.getByAltText("shot.png")).toBeInTheDocument();
    expect(textarea().value).toBe("look at this");
  });

  it("removing a given-back image keeps the thumbnail the failed message still shows", async () => {
    failStreamWith("provider exploded");
    const { container } = render(<ChatPanel persona="nell" />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [new File(["x"], "shot.png", { type: "image/png" })] } });
    });
    await waitFor(() => expect(screen.getByAltText("shot.png")).toBeInTheDocument());
    await typeAndSend("look at this");
    await waitFor(() => expect(screen.getByRole("button", { name: "remove file" })).toBeInTheDocument());
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "remove file" }));
    });
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith("blob:test-1");
  });
});
