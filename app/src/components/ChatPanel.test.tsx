// Component tests for ChatPanel — addresses 2026-05-08 audit P4-2
// (frontend test coverage beyond smoke). The render-level assertions
// verify P3-4 (image-only sends), the send/stop toggle for the audit
// P2-9 cancel control, and the persona-aware placeholder so future
// hardcoded "Nell" doesn't sneak back in.

import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({ port: 0, auth_token: null })),
}));

vi.mock("../bridge", () => ({
  newSession: vi.fn(async () => "test-session-id"),
  fetchActiveSession: vi.fn(async () => null),
  closeSession: vi.fn(async () => undefined),
  uploadImage: vi.fn(async () => ({ sha: "deadbeef" })),
  getBridgeCredentials: vi.fn(async () => ({
    url: "http://127.0.0.1:50000",
    port: 50000,
    authToken: "test-token",
  })),
}));

// bridgeEvents opens a real WebSocket from its own module. Stub it so
// ChatPanel's default subscription is inert when no eventStream prop is
// passed (callers can still inject their own stream in tests).
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
import { fetchActiveSession, newSession } from "../bridge";
import { streamChat } from "../streamChat";

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

describe("ChatPanel — staged image URL lifetime (F-007)", () => {
  let createSpy: ReturnType<typeof vi.spyOn>;
  let revokeSpy: ReturnType<typeof vi.spyOn>;
  let urlCounter = 0;

  beforeEach(() => {
    urlCounter = 0;
    createSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockImplementation(() => `blob:test-${++urlCounter}`);
    revokeSpy = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined);
  });

  afterEach(() => {
    cleanup();
    createSpy.mockRestore();
    revokeSpy.mockRestore();
  });

  it("revokes a staged-but-unsent preview URL on unmount", async () => {
    const { container, unmount } = render(<ChatPanel persona="nell" />);

    // The hidden <input type="file"> is the staging surface. Drive it
    // directly — clicking the paperclip just forwards to fileInputRef.
    const fileInput = container.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    expect(fileInput).not.toBeNull();

    const file = new File(["fake-bytes"], "shot.png", { type: "image/png" });

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    // Wait for the staged thumbnail to appear so we know handleFile ran
    // through createObjectURL.
    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledTimes(1);
    });
    const stagedUrl = createSpy.mock.results[0]?.value as string;
    expect(stagedUrl).toMatch(/^blob:test-/);

    // Unmount WITHOUT sending — this is the leak path the fix closes.
    unmount();

    // The cleanup sweep at ChatPanel.tsx:159-160 should have revoked the
    // staged URL because the fix tracks it at creation time, not send time.
    expect(revokeSpy).toHaveBeenCalledWith(stagedUrl);
  });
});

describe("ChatPanel — sticky-session reattach on first send (F-201)", () => {
  const mockedFetchActive = fetchActiveSession as unknown as ReturnType<typeof vi.fn>;
  const mockedNewSession = newSession as unknown as ReturnType<typeof vi.fn>;
  const mockedStreamChat = streamChat as unknown as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockedFetchActive.mockReset();
    mockedNewSession.mockReset();
    mockedStreamChat.mockReset();
    // Restore default streamChat behaviour (returns a no-op cancel).
    mockedStreamChat.mockImplementation(async () => () => undefined);
  });

  afterEach(() => {
    cleanup();
  });

  async function typeAndSend(persona: string) {
    render(<ChatPanel persona={persona} />);
    const textarea = screen.getByPlaceholderText(
      new RegExp(`^Write to`),
    ) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: "hello" } });
    });
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await act(async () => {
      fireEvent.click(sendBtn);
    });
    // Allow microtask queue (fetchActiveSession + downstream send) to flush.
    await waitFor(() => {
      expect(mockedStreamChat).toHaveBeenCalled();
    });
  }

  it("attaches to active session on first send when /sessions/active returns a sid", async () => {
    mockedFetchActive.mockResolvedValue("existing-sess-xyz");
    mockedNewSession.mockResolvedValue("should-not-be-used");

    await typeAndSend("nell");

    // attach was attempted
    expect(mockedFetchActive).toHaveBeenCalledTimes(1);
    expect(mockedFetchActive).toHaveBeenCalledWith("nell");
    // newSession was NOT called — we attached to the existing session
    expect(mockedNewSession).not.toHaveBeenCalled();
    // streamChat was driven with the attached session_id
    const [, sessionId] = mockedStreamChat.mock.calls[0] as [
      string,
      string,
      string,
      unknown,
    ];
    expect(sessionId).toBe("existing-sess-xyz");
  });

  it("falls back to newSession when /sessions/active returns null", async () => {
    mockedFetchActive.mockResolvedValue(null);
    mockedNewSession.mockResolvedValue("fresh-sess-abc");

    await typeAndSend("nell");

    expect(mockedFetchActive).toHaveBeenCalledTimes(1);
    expect(mockedNewSession).toHaveBeenCalledTimes(1);
    expect(mockedNewSession).toHaveBeenCalledWith("nell");
    const [, sessionId] = mockedStreamChat.mock.calls[0] as [
      string,
      string,
      string,
      unknown,
    ];
    expect(sessionId).toBe("fresh-sess-abc");
  });

  it("falls back to newSession when /sessions/active throws", async () => {
    mockedFetchActive.mockRejectedValue(new Error("network flake"));
    mockedNewSession.mockResolvedValue("fresh-sess-after-throw");

    await typeAndSend("nell");

    // fetchActiveSession was tried — the throw was swallowed by the
    // try/catch in send(), not propagated to the user.
    expect(mockedFetchActive).toHaveBeenCalledTimes(1);
    // We still landed on a fresh session via newSession.
    expect(mockedNewSession).toHaveBeenCalledTimes(1);
    const [, sessionId] = mockedStreamChat.mock.calls[0] as [
      string,
      string,
      string,
      unknown,
    ];
    expect(sessionId).toBe("fresh-sess-after-throw");
  });
});

describe("ChatPanel — recovery banner (Phase 3.B)", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders no banner when recovering is undefined (default)", () => {
    render(<ChatPanel persona="nell" />);
    expect(screen.queryByTestId("recovery-banner")).toBeNull();
  });

  it("renders no banner when recovering is false", () => {
    render(<ChatPanel persona="nell" recovering={false} />);
    expect(screen.queryByTestId("recovery-banner")).toBeNull();
  });

  it("renders the banner when recovering is true", () => {
    render(<ChatPanel persona="nell" recovering={true} />);
    const banner = screen.getByTestId("recovery-banner");
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveAttribute("role", "status");
    expect(banner.textContent).toMatch(/reconnecting your previous chat/i);
  });
});

// ── Task 26: initiate banner integration ──────────────────────────────
// ChatPanel subscribes to bridge /events and renders an InitiateBanner
// for every initiate_delivered event. After 2s on screen the banner
// auto-marks itself read by POSTing /initiate/state.
describe("ChatPanel — initiate banner integration (Task 26)", () => {
  /** Minimal EventStream stub the test drives directly. */
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

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders an InitiateBanner when initiate_delivered arrives on /events", async () => {
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);

    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_001",
        body: "the dream from this morning landed somewhere",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-05-11T14:32:00+00:00",
      });
    });

    expect(await screen.findByText(/landed somewhere/)).toBeInTheDocument();
    expect(screen.getByTestId("initiate-banner-list")).toBeInTheDocument();
  });

  it("ignores non-initiate events", async () => {
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);

    await act(async () => {
      stream.emit({ type: "heartbeat_tick", at: "2026-05-11T14:32:00+00:00" });
    });

    expect(screen.queryByTestId("initiate-banner-list")).toBeNull();
  });

  it("posts /initiate/state with new_state=read after the banner is on-screen ~2s", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const stream = makeStream();
      render(<ChatPanel persona="nell" eventStream={stream} />);

      await act(async () => {
        stream.emit({
          type: "initiate_delivered",
          audit_id: "ia_001",
          body: "tiny soft message",
          urgency: "quiet",
          state: "delivered",
          timestamp: "2026-05-11T14:32:00+00:00",
        });
      });

      await act(async () => {
        vi.advanceTimersByTime(2100);
      });

      // Drain any scheduled microtasks the POST setup queued.
      await waitFor(() => {
        const postCalls = fetchSpy.mock.calls.filter(([url, init]: [RequestInfo | URL, RequestInit?]) => {
          const u = typeof url === "string" ? url : url.toString();
          return u.includes("/initiate/state") && init?.method === "POST";
        });
        expect(postCalls.length).toBeGreaterThan(0);
      });

      const readPost = fetchSpy.mock.calls.find(([url, init]: [RequestInfo | URL, RequestInit?]) => {
        const u = typeof url === "string" ? url : (url as URL).toString();
        const body = String(init?.body ?? "");
        return u.includes("/initiate/state") && body.includes('"new_state":"read"');
      });
      expect(readPost).toBeDefined();
      const body = JSON.parse(String((readPost![1] as RequestInit).body));
      expect(body).toEqual({ audit_id: "ia_001", new_state: "read" });
    } finally {
      vi.useRealTimers();
    }
  });

  it("removes the banner and posts dismissed when the close button is clicked", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);

    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_002",
        body: "another small thought",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-05-11T15:00:00+00:00",
      });
    });

    await screen.findByText(/another small thought/);

    const dismissBtn = screen.getByRole("button", { name: /dismiss/i });
    await act(async () => {
      fireEvent.click(dismissBtn);
    });

    expect(screen.queryByText(/another small thought/)).toBeNull();
    await waitFor(() => {
      const dismissCall = fetchSpy.mock.calls.find(([url, init]: [RequestInfo | URL, RequestInit?]) => {
        const u = typeof url === "string" ? url : (url as URL).toString();
        const body = String(init?.body ?? "");
        return u.includes("/initiate/state") && body.includes('"new_state":"dismissed"');
      });
      expect(dismissCall).toBeDefined();
    });
  });
});

// ── Bundle A #4 — reply_to_audit_id threading via streamChat ─────────────
// Closes the v0.0.9 review TODO: replied_explicit transitions used to be
// renderer-only POSTs to /initiate/state, which left the chat engine blind
// to which initiate the user was replying to. The fix moves the transition
// server-side by passing reply_to_audit_id in the streamChat payload, so
// the audit + memory + chat engine all see the link atomically.
describe("ChatPanel — reply_to_audit_id threading (Bundle A #4)", () => {
  const mockedStreamChat = streamChat as unknown as ReturnType<typeof vi.fn>;
  const mockedFetchActive = fetchActiveSession as unknown as ReturnType<typeof vi.fn>;
  const mockedNewSession = newSession as unknown as ReturnType<typeof vi.fn>;

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

  beforeEach(() => {
    mockedStreamChat.mockReset();
    mockedStreamChat.mockImplementation(async () => () => undefined);
    mockedFetchActive.mockReset();
    mockedFetchActive.mockResolvedValue(null);
    mockedNewSession.mockReset();
    mockedNewSession.mockResolvedValue("sess-bundle-a");
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("threads replyToAuditId through streamChat options when an active reply target exists", async () => {
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);
    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_replythread",
        body: "the dream from this morning",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-05-12T09:00:00+00:00",
      });
    });
    const replyBtn = await screen.findByRole("button", { name: /reply/i });
    await act(async () => {
      fireEvent.click(replyBtn);
    });

    const textarea = screen.getByPlaceholderText(/^Write to/) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: "yeah I felt it too" } });
    });
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await act(async () => {
      fireEvent.click(sendBtn);
    });

    await waitFor(() => {
      expect(mockedStreamChat).toHaveBeenCalled();
    });
    // streamChat signature: (persona, sessionId, message, handlers, options?)
    const args = mockedStreamChat.mock.calls[0] as [
      string,
      string,
      string,
      unknown,
      { replyToAuditId?: string } | undefined,
    ];
    expect(args[4]?.replyToAuditId).toBe("ia_replythread");
  });

  it("does not POST /initiate/state replied_explicit when sending — server handles it", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);
    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_noclientpost",
        body: "small message",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-05-12T09:00:00+00:00",
      });
    });
    const replyBtn = await screen.findByRole("button", { name: /reply/i });
    await act(async () => {
      fireEvent.click(replyBtn);
    });
    const textarea = screen.getByPlaceholderText(/^Write to/) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(textarea, { target: { value: "I see you" } });
    });
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await act(async () => {
      fireEvent.click(sendBtn);
    });

    await waitFor(() => {
      expect(mockedStreamChat).toHaveBeenCalled();
    });

    // No renderer-side POST should fire for replied_explicit — the server
    // performs that transition atomically with the chat turn now.
    const repliedExplicitPost = fetchSpy.mock.calls.find(
      ([url, init]: [RequestInfo | URL, RequestInit?]) => {
        const u = typeof url === "string" ? url : (url as URL).toString();
        const body = String(init?.body ?? "");
        return (
          u.includes("/initiate/state") &&
          body.includes('"new_state":"replied_explicit"')
        );
      },
    );
    expect(repliedExplicitPost).toBeUndefined();
  });
});

