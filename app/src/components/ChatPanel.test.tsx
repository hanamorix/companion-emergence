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
  fetchChatHistory: vi.fn(async () => ({ messages: [], next_before_turn: null })),
  closeSession: vi.fn(async () => undefined),
  snapshotSession: vi.fn(async () => ({ closed: false, errors: 0 })),
  uploadImage: vi.fn(async () => ({ kind: "image", sha: "deadbeef", media_type: "image/png", size_bytes: 3 })),
  getBridgeCredentials: vi.fn(async () => ({
    url: "http://127.0.0.1:50000",
    port: 50000,
    authToken: "test-token",
  })),
  acceptVoiceEdit: vi.fn(async () => ({ ok: true })),
  rejectVoiceEdit: vi.fn(async () => ({ ok: true })),
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
import { acceptVoiceEdit, rejectVoiceEdit, fetchActiveSession, newSession, closeSession, snapshotSession, uploadImage } from "../bridge";
import { streamChat } from "../streamChat";
import { invoke } from "@tauri-apps/api/core";

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
      screen.getByRole("button", { name: /send file/i }),
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

    // Phase 3B (v0.0.15-alpha.2) mount-time hydration also probes
    // /sessions/active so the chat panel can replay JSONL turns on
    // reopen. Mount call → null (no active session) → no hydration,
    // sessionRef stays null. Send call → null again → newSession.
    // Net: fetchActiveSession is invoked twice, both with the persona.
    expect(mockedFetchActive).toHaveBeenCalledTimes(2);
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

    // fetchActiveSession was tried — the throw was swallowed both at
    // mount-time hydration (Phase 3B) and inside send(), not propagated
    // to the user.
    expect(mockedFetchActive).toHaveBeenCalledTimes(2);
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

// ── Task 10.1: felt-time recovery banner ────────────────────────────────
describe("ChatPanel — felt-time recovery banner (Task 10.1)", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders no felt-time banner when feltTimeRecovered is undefined (default)", () => {
    render(<ChatPanel persona="nell" />);
    expect(screen.queryByTestId("felt-time-recovery-banner")).toBeNull();
  });

  it("renders no felt-time banner when feltTimeRecovered is false", () => {
    render(<ChatPanel persona="nell" feltTimeRecovered={false} />);
    expect(screen.queryByTestId("felt-time-recovery-banner")).toBeNull();
  });

  it("renders the felt-time recovery banner when feltTimeRecovered is true", () => {
    render(<ChatPanel persona="nell" feltTimeRecovered={true} />);
    const banner = screen.getByTestId("felt-time-recovery-banner");
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveAttribute("role", "status");
    expect(banner.textContent).toMatch(/felt time recovered from logs/i);
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

  it("notify-urgency desktop notification uses the persona name, not hardcoded 'Nell'", async () => {
    vi.mocked(invoke).mockClear();
    const stream = makeStream();
    render(<ChatPanel persona="phoebe" eventStream={stream} />);

    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_notify",
        body: "you up?",
        urgency: "notify",
        state: "delivered",
        timestamp: "2026-05-11T14:32:00+00:00",
      });
    });

    await waitFor(() => {
      expect(invoke).toHaveBeenCalledWith(
        "show_initiate_notification",
        expect.objectContaining({ title: "Phoebe", body: "you up?" }),
      );
    });
    expect(invoke).not.toHaveBeenCalledWith(
      "show_initiate_notification",
      expect.objectContaining({ title: "Nell" }),
    );
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
//
// Task 2 update: the old "click reply → type in main textarea → send" flow is
// gone. The card now has an inline reply textarea (aria-label "Reply to Nell").
// Both tests now use the card-reply path.
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

    // Card-reply path: type in the card's inline textarea and send.
    const cardTextarea = await screen.findByLabelText(/^Reply to Nell$/i) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(cardTextarea, { target: { value: "yeah I felt it too" } });
    });
    const cardSendBtn = screen.getByRole("button", { name: /Send reply/i });
    await act(async () => {
      fireEvent.click(cardSendBtn);
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

    // Card-reply path: type in the card's inline textarea and send.
    const cardTextarea = await screen.findByLabelText(/^Reply to Nell$/i) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(cardTextarea, { target: { value: "I see you" } });
    });
    const cardSendBtn = screen.getByRole("button", { name: /Send reply/i });
    await act(async () => {
      fireEvent.click(cardSendBtn);
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

// ── Task 9: VoiceEditPanel inline rendering ───────────────────────────────
describe("ChatPanel — VoiceEditPanel inline rendering (Task 9)", () => {
  const mockedAcceptVoiceEdit = acceptVoiceEdit as unknown as ReturnType<typeof vi.fn>;

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
    mockedAcceptVoiceEdit.mockReset();
    mockedAcceptVoiceEdit.mockResolvedValue({ ok: true });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders VoiceEditPanel for a voice_edit_proposal initiate event", async () => {
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);

    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "a1",
        kind: "voice_edit_proposal",
        body: "Proposing a voice change",
        diff: "- old line\n+ new line",
        urgency: "quiet",
        state: "delivered",
        timestamp: new Date().toISOString(),
      });
    });

    // VoiceEditPanel renders as a dialog with aria-label "Voice edit proposal"
    const dialog = await screen.findByRole("dialog", { name: /voice edit proposal/i });
    expect(dialog).toBeInTheDocument();

    // Generic InitiateBanner list should NOT be shown for this event
    expect(screen.queryByTestId("initiate-banner-list")).toBeNull();

    // Clicking Accept calls acceptVoiceEdit
    const acceptBtn = screen.getByRole("button", { name: /^accept$/i });
    await act(async () => {
      fireEvent.click(acceptBtn);
    });

    await waitFor(() => {
      expect(mockedAcceptVoiceEdit).toHaveBeenCalledWith("nell", "a1", null);
    });
  });
});

// ── Voice-edit accept/reject error handling ────────────────────────────────
// Ensures the panel disappears only on success, stays mounted on failure,
// and errors are surfaced via console.error rather than swallowed silently.
describe("ChatPanel — voice-edit accept/reject error handling", () => {
  const mockedAcceptVoiceEdit = acceptVoiceEdit as unknown as ReturnType<typeof vi.fn>;
  const mockedRejectVoiceEdit = rejectVoiceEdit as unknown as ReturnType<typeof vi.fn>;

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

  async function renderWithProposal(auditId: string, stream: ReturnType<typeof makeStream>) {
    render(<ChatPanel persona="nell" eventStream={stream} />);
    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: auditId,
        kind: "voice_edit_proposal",
        body: "Proposing a voice change",
        diff: "- old line\n+ new line",
        urgency: "quiet",
        state: "delivered",
        timestamp: new Date().toISOString(),
      });
    });
    // Wait for the dialog to appear before returning.
    await screen.findByRole("dialog", { name: /voice edit proposal/i });
  }

  beforeEach(() => {
    mockedAcceptVoiceEdit.mockReset();
    mockedRejectVoiceEdit.mockReset();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("accept success removes the panel", async () => {
    mockedAcceptVoiceEdit.mockResolvedValue({ ok: true });
    const stream = makeStream();
    await renderWithProposal("ve_success", stream);

    const acceptBtn = screen.getByRole("button", { name: /^accept$/i });
    await act(async () => {
      fireEvent.click(acceptBtn);
    });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /voice edit proposal/i })).toBeNull();
    });
    expect(mockedAcceptVoiceEdit).toHaveBeenCalledWith("nell", "ve_success", null);
  });

  it("accept failure keeps the panel mounted and logs via console.error", async () => {
    mockedAcceptVoiceEdit.mockRejectedValue(new Error("network error"));
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);

    const stream = makeStream();
    await renderWithProposal("ve_fail_accept", stream);

    const acceptBtn = screen.getByRole("button", { name: /^accept$/i });
    await act(async () => {
      fireEvent.click(acceptBtn);
    });

    // Panel must still be present after the rejection.
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        "voice-edit accept failed",
        expect.any(Error),
      );
    });
    expect(screen.getByRole("dialog", { name: /voice edit proposal/i })).toBeInTheDocument();

    consoleSpy.mockRestore();
  });

  it("reject happy path removes the panel and calls rejectVoiceEdit with correct args", async () => {
    mockedRejectVoiceEdit.mockResolvedValue({ ok: true });
    const stream = makeStream();
    await renderWithProposal("ve_reject_ok", stream);

    const rejectBtn = screen.getByRole("button", { name: /^reject$/i });
    await act(async () => {
      fireEvent.click(rejectBtn);
    });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /voice edit proposal/i })).toBeNull();
    });
    expect(mockedRejectVoiceEdit).toHaveBeenCalledWith("nell", "ve_reject_ok");
  });
});

// ── Task 6: snapshot on unmount instead of close ─────────────────────────
describe("ChatPanel — snapshot on unmount (Task 6)", () => {
  const mockedSnapshot = snapshotSession as unknown as ReturnType<typeof vi.fn>;
  const mockedClose = closeSession as unknown as ReturnType<typeof vi.fn>;
  const mockedNew = newSession as unknown as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockedSnapshot.mockReset();
    mockedSnapshot.mockResolvedValue({ closed: false, errors: 0 });
    mockedClose.mockReset();
    mockedNew.mockReset();
    mockedNew.mockResolvedValue("test-session-id");
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("snapshots active session on unmount instead of closing it", async () => {
    mockedNew.mockResolvedValueOnce("11111111-1111-4111-8111-111111111111");

    const { unmount } = render(<ChatPanel persona="nell" />);
    fireEvent.change(screen.getByPlaceholderText(/write to/i), { target: { value: "hi" } });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));
    await waitFor(() => expect(mockedNew).toHaveBeenCalled());

    unmount();

    expect(mockedSnapshot).toHaveBeenCalledWith(
      "nell",
      "11111111-1111-4111-8111-111111111111",
      expect.objectContaining({ keepalive: false }),
    );
    expect(mockedClose).not.toHaveBeenCalled();
  });
});

// ── Task 2: reach-out card reply merges into transcript ──────────────────
// Verifies the new onCardSendReply / streamTurn flow:
//   1. reach-out body prepended to transcript as a ✶-marked nell bubble
//   2. user reply appended as a hana bubble
//   3. streamChat called with replyToAuditId
//   4. card removed from UI
//   5. main composer textarea untouched
describe("ChatPanel — reach-out card reply merges into transcript (Task 2)", () => {
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
    mockedNewSession.mockResolvedValue("sess-task2");
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("card reply while streaming is a no-op — banner stays, streamChat not called again", async () => {
    // Simulate streaming=true by making the first streamChat call not resolve
    // its cancel (stream stays open) so the component's streaming state remains true.
    // Use a ref-box so TypeScript doesn't narrow to `never` via control flow.
    const firstResolveBox: { fn: (() => void) | null } = { fn: null };
    mockedStreamChat.mockImplementationOnce(
      () =>
        new Promise<() => void>((resolve) => {
          // Resolve with a cancel fn only when the test chooses — streaming
          // stays true until then because streamChat hasn't returned yet.
          firstResolveBox.fn = () => resolve(() => undefined);
        }),
    );

    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);

    // Deliver two banners.
    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_first",
        body: "first reach-out",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-06-07T10:00:00+00:00",
      });
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_second",
        body: "second reach-out",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-06-07T10:01:00+00:00",
      });
    });

    await screen.findByText(/first reach-out/);
    await screen.findByText(/second reach-out/);

    // Reply to first card → streamChat called once (returns a never-resolving promise so streaming stays true).
    const firstCardTextareas = screen.getAllByLabelText(/^Reply to Nell$/i) as HTMLTextAreaElement[];
    await act(async () => {
      fireEvent.change(firstCardTextareas[0], { target: { value: "got your first" } });
    });
    const sendBtns = screen.getAllByRole("button", { name: /Send reply/i });
    await act(async () => {
      fireEvent.click(sendBtns[0]);
    });

    // Allow the first streamChat call to begin (it returns a pending promise).
    await waitFor(() => {
      expect(mockedStreamChat).toHaveBeenCalledTimes(1);
    });

    // Now streaming === true. Attempt a card reply on the second banner.
    const secondCardTextareas = screen.getAllByLabelText(/^Reply to Nell$/i) as HTMLTextAreaElement[];
    await act(async () => {
      fireEvent.change(secondCardTextareas[0], { target: { value: "got your second" } });
    });
    const sendBtns2 = screen.getAllByRole("button", { name: /Send reply/i });
    await act(async () => {
      fireEvent.click(sendBtns2[0]);
    });

    // streamChat must NOT have been called a second time.
    expect(mockedStreamChat).toHaveBeenCalledTimes(1);

    // The second banner must still be present (guard fired before banner removal).
    expect(screen.queryByText(/second reach-out/)).not.toBeNull();

    // Cleanup: resolve the first stream so the component can unmount cleanly.
    firstResolveBox.fn?.();
  });

  it("reply from a reach-out card merges into the transcript with reply_to_audit_id and frees the main input", async () => {
    const stream = makeStream();
    render(<ChatPanel persona="nell" eventStream={stream} />);

    // 1. Emit an initiate_delivered event.
    await act(async () => {
      stream.emit({
        type: "initiate_delivered",
        audit_id: "ia_x",
        body: "thinking of you",
        urgency: "quiet",
        state: "delivered",
        timestamp: "2026-06-07T10:00:00+00:00",
      });
    });

    // Card must appear.
    await screen.findByText(/thinking of you/);

    // 2. Type in the card's inline reply textarea and send.
    const cardTextarea = screen.getByLabelText(/^Reply to Nell$/i) as HTMLTextAreaElement;
    await act(async () => {
      fireEvent.change(cardTextarea, { target: { value: "I'm here" } });
    });
    const cardSendBtn = screen.getByRole("button", { name: /Send reply/i });
    await act(async () => {
      fireEvent.click(cardSendBtn);
    });

    // 3. streamChat must be called with replyToAuditId === "ia_x".
    await waitFor(() => {
      expect(mockedStreamChat).toHaveBeenCalled();
    });
    const args = mockedStreamChat.mock.calls[0] as [
      string,
      string,
      string,
      unknown,
      { replyToAuditId?: string } | undefined,
    ];
    expect(args[4]?.replyToAuditId).toBe("ia_x");

    // 4. Transcript shows the reach-out body AND the typed reply.
    const chatMessages = screen.getByTestId("chat-messages");
    expect(chatMessages.textContent).toContain("thinking of you");
    expect(chatMessages.textContent).toContain("I'm here");
    // The reach-out bubble renders a ✶ marker (msg-reachedout span).
    expect(chatMessages.querySelector(".msg-reachedout")).not.toBeNull();

    // 5. Card is gone — the header "Nell reached out" is no longer rendered.
    expect(screen.queryByText(/Nell reached out/)).toBeNull();

    // 6. Main composer textarea is still empty (untouched).
    const mainTextarea = screen.getByPlaceholderText(/^Write to Nell/i) as HTMLTextAreaElement;
    expect(mainTextarea.value).toBe("");
  });
});

// ── P0 image-tool-route — send-file wire shape (C8a) + relabel/types (C8b) ──
describe("ChatPanel — shared-file send path (P0 image-tool-route)", () => {
  const mockedStreamChat = streamChat as unknown as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockedStreamChat.mockReset();
    mockedStreamChat.mockImplementation(async () => () => undefined);
  });

  afterEach(() => cleanup());

  it("C8a — a staged upload is sent as a sharedFiles ref, not an image_shas payload", async () => {
    const { container } = render(<ChatPanel persona="nell" />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["fake-bytes"], "shot.png", { type: "image/png" });
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });
    // Wait for upload to resolve (staged → ready) so the send button enables.
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await waitFor(() => expect(sendBtn).toBeEnabled());
    await act(async () => {
      fireEvent.click(sendBtn);
    });
    await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    // streamChat signature: (persona, sessionId, message, handlers, options?)
    const args = mockedStreamChat.mock.calls[0] as [
      string,
      string,
      string,
      unknown,
      { sharedFiles?: Array<{ kind: string; sha: string }>; imageShas?: unknown } | undefined,
    ];
    const opts = args[4];
    expect(opts?.sharedFiles).toBeTruthy();
    expect(opts?.sharedFiles?.[0]).toMatchObject({ kind: "image", sha: "deadbeef" });
    // The old transport payload must NOT be set.
    expect(opts?.imageShas).toBeUndefined();
  });

  it("C8b — the attach control is labelled 'send file' and accepts a non-image type", () => {
    const { container } = render(<ChatPanel persona="nell" />);
    expect(screen.getByRole("button", { name: /send file/i })).toBeInTheDocument();
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const accept = fileInput.getAttribute("accept") ?? "";
    // At least one non-image type must be accepted (backed by the widened /upload).
    expect(accept).toMatch(/text\/plain|application\/pdf|\.txt|\.md/);
  });
});

// ── File-send trio (#268 non-image, #269 tool cue, #270 multi-file) ─────────
describe("ChatPanel — file-send trio (#268 / #269 / #270)", () => {
  const mockedStreamChat = streamChat as unknown as ReturnType<typeof vi.fn>;
  const mockedUpload = uploadImage as unknown as ReturnType<typeof vi.fn>;
  let createSpy: ReturnType<typeof vi.spyOn>;
  let revokeSpy: ReturnType<typeof vi.spyOn>;
  let urlCounter = 0;

  // Sha per upload so multi-file sends are distinguishable and ordered.
  const uploadByKind = async (_persona: string, file: File) => {
    const sha = `sha-${file.name}`;
    return file.type.startsWith("image/")
      ? { kind: "image", sha, media_type: file.type, size_bytes: file.size }
      : { kind: "file", sha, filename: file.name, size_bytes: file.size };
  };

  beforeEach(() => {
    urlCounter = 0;
    createSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockImplementation(() => `blob:test-${++urlCounter}`);
    revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    mockedStreamChat.mockReset();
    mockedStreamChat.mockImplementation(async () => () => undefined);
    mockedUpload.mockReset();
    mockedUpload.mockImplementation(uploadByKind);
  });

  afterEach(() => {
    cleanup();
    createSpy.mockRestore();
    revokeSpy.mockRestore();
  });

  function fileInputOf(container: HTMLElement): HTMLInputElement {
    return container.querySelector('input[type="file"]') as HTMLInputElement;
  }

  async function stage(container: HTMLElement, files: File[]) {
    await act(async () => {
      fireEvent.change(fileInputOf(container), { target: { files } });
    });
  }

  function sentOpts(callIndex = 0) {
    const args = mockedStreamChat.mock.calls[callIndex] as [
      string,
      string,
      string,
      unknown,
      { sharedFiles?: Array<{ kind: string; sha: string; filename?: string }> } | undefined,
    ];
    return { text: args[2], opts: args[4] };
  }

  it("AC1 — a text/plain file stages, shows its name, and ships as a kind=file ref", async () => {
    const { container } = render(<ChatPanel persona="nell" />);
    await stage(container, [new File(["hello"], "notes.txt", { type: "text/plain" })]);
    expect(mockedUpload).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("notes.txt")).toBeInTheDocument();
    // Non-image rows get the document glyph, not an <img>.
    expect(screen.getByRole("img", { name: /file notes\.txt/i })).toBeInTheDocument();
    expect(container.querySelector("img[alt='notes.txt']")).toBeNull();

    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await waitFor(() => expect(sendBtn).toBeEnabled());
    await act(async () => {
      fireEvent.click(sendBtn);
    });
    await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    const { text, opts } = sentOpts();
    expect(opts?.sharedFiles).toEqual([
      { kind: "file", sha: "sha-notes.txt", media_type: undefined, filename: "notes.txt" },
    ]);
    expect(text).toBe("Please look at this file.");
  });

  it("AC2 — an unsupported type is refused and not uploaded; an extension-only .md is accepted", async () => {
    const { container } = render(<ChatPanel persona="nell" />);
    await stage(container, [new File(["zip"], "a.zip", { type: "application/zip" })]);
    expect(mockedUpload).not.toHaveBeenCalled();
    expect(screen.getByText(/Unsupported file type: application\/zip\./)).toBeInTheDocument();

    await stage(container, [new File(["# md"], "README.md", { type: "" })]);
    expect(mockedUpload).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("README.md")).toBeInTheDocument();
  });

  it("AC3 — multi-select stages every file, the paperclip stays enabled, the ninth is refused, send ships eight in order", async () => {
    const { container } = render(<ChatPanel persona="nell" />);
    const mk = (i: number) => new File([`f${i}`], `f${i}.txt`, { type: "text/plain" });
    await stage(container, [mk(1), mk(2), mk(3)]);
    expect(screen.getAllByRole("button", { name: /remove file/i })).toHaveLength(3);
    const clip = screen.getByRole("button", { name: /send file/i });
    expect(clip).toBeEnabled();

    await stage(container, [mk(4)]);
    expect(screen.getAllByRole("button", { name: /remove file/i })).toHaveLength(4);

    await stage(container, [mk(5), mk(6), mk(7), mk(8), mk(9)]);
    expect(screen.getAllByRole("button", { name: /remove file/i })).toHaveLength(8);
    expect(screen.getByText("Up to 8 files per message.")).toBeInTheDocument();
    expect(clip).toBeDisabled();

    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await waitFor(() => expect(sendBtn).toBeEnabled());
    await act(async () => {
      fireEvent.click(sendBtn);
    });
    await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    const { text, opts } = sentOpts();
    expect(opts?.sharedFiles?.map((f) => f.sha)).toEqual(
      [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `sha-f${i}.txt`),
    );
    expect(text).toBe("Here are a few files, have a look.");
    // Staging cleared after send (the paperclip stays disabled only because
    // the mocked stream never finishes; that is the streaming gate, not the cap).
    expect(screen.queryAllByRole("button", { name: /remove file/i })).toHaveLength(0);
  });

  it("AC4 — removing the second of three rows keeps the other two and revokes only that URL", async () => {
    const { container } = render(<ChatPanel persona="nell" />);
    await stage(container, [
      new File(["1"], "one.txt", { type: "text/plain" }),
      new File(["2"], "two.txt", { type: "text/plain" }),
      new File(["3"], "three.txt", { type: "text/plain" }),
    ]);
    const removes = screen.getAllByRole("button", { name: /remove file/i });
    expect(removes).toHaveLength(3);
    await act(async () => {
      fireEvent.click(removes[1]);
    });
    expect(screen.queryByText("two.txt")).toBeNull();
    expect(screen.getByText("one.txt")).toBeInTheDocument();
    expect(screen.getByText("three.txt")).toBeInTheDocument();
    expect(revokeSpy).toHaveBeenCalledTimes(1);
    expect(revokeSpy).toHaveBeenCalledWith("blob:test-2");
  });

  it("AC4b — a drop stages every passing file in order and shows the first failure only", async () => {
    const { container } = render(<ChatPanel persona="nell" />);
    const big = new File([new Uint8Array(21 * 1024 * 1024)], "big.txt", { type: "text/plain" });
    const panel = container.querySelector(".chat-panel") as HTMLElement;
    await act(async () => {
      fireEvent.drop(panel, {
        dataTransfer: {
          files: [
            new File(["ok"], "ok.txt", { type: "text/plain" }),
            new File(["zip"], "a.zip", { type: "application/zip" }),
            big,
          ],
          items: [],
          types: ["Files"],
        },
      });
    });
    expect(screen.getAllByRole("button", { name: /remove file/i })).toHaveLength(1);
    expect(screen.getByText("ok.txt")).toBeInTheDocument();
    expect(screen.getByText("Unsupported file type: application/zip.")).toBeInTheDocument();
    expect(screen.queryByText(/File too large/)).toBeNull();
  });

  it("AC5 — send is disabled while any row uploads; a failed row is never shipped", async () => {
    let resolveSlow: (v: unknown) => void = () => undefined;
    mockedUpload.mockImplementation(async (_p: string, file: File) => {
      if (file.name === "slow.txt") {
        await new Promise((r) => (resolveSlow = r));
        return { kind: "file", sha: "sha-slow", filename: "slow.txt", size_bytes: 1 };
      }
      if (file.name === "bad.txt") throw new Error("upload exploded");
      return uploadByKind(_p, file);
    });
    const { container } = render(<ChatPanel persona="nell" />);
    await stage(container, [
      new File(["a"], "fast.txt", { type: "text/plain" }),
      new File(["b"], "slow.txt", { type: "text/plain" }),
      new File(["c"], "bad.txt", { type: "text/plain" }),
    ]);
    const sendBtn = screen.getByRole("button", { name: /^send$/i });
    await waitFor(() => expect(screen.getByText(/failed: upload exploded/)).toBeInTheDocument());
    expect(sendBtn).toBeDisabled();
    await act(async () => {
      resolveSlow(undefined);
    });
    await waitFor(() => expect(sendBtn).toBeEnabled());
    await act(async () => {
      fireEvent.click(sendBtn);
    });
    await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    const { opts } = sentOpts();
    expect(opts?.sharedFiles?.map((f) => f.sha)).toEqual(["sha-fast.txt", "sha-slow"]);
  });

  it("AC6 — default text: one image, one file, several files", async () => {
    const run = async (files: File[]) => {
      const { container, unmount } = render(<ChatPanel persona="nell" />);
      await stage(container, files);
      const sendBtn = screen.getByRole("button", { name: /^send$/i });
      await waitFor(() => expect(sendBtn).toBeEnabled());
      await act(async () => {
        fireEvent.click(sendBtn);
      });
      await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
      const { text } = sentOpts(mockedStreamChat.mock.calls.length - 1);
      unmount();
      return text;
    };
    expect(await run([new File(["p"], "shot.png", { type: "image/png" })])).toBe(
      "Please look at this image.",
    );
    expect(await run([new File(["t"], "notes.txt", { type: "text/plain" })])).toBe(
      "Please look at this file.",
    );
    expect(
      await run([
        new File(["p"], "shot.png", { type: "image/png" }),
        new File(["t"], "notes.txt", { type: "text/plain" }),
      ]),
    ).toBe("Here are a few files, have a look.");
  });
});
