/**
 * WS streaming chat client, talks to /stream/{session_id}.
 *
 * Wire protocol (from brain/bridge/server.py):
 *   client -> server: {message: string, image_shas?: string[],
 *                      reply_to_audit_id?: string} after WS open
 *   server -> client:
 *     {type: "started", session_id, at}
 *     {type: "tool_call", tool, session_id, at}
 *     {type: "tool_result", tool, summary, ...}
 *     {type: "reply_chunk", text}
 *     {type: "done", session_id, turn, duration_ms, metadata, at}
 *   on error:
 *     {type: "error", code, detail?, done: true}
 */

import { getBridgeCredentials, resetBridgeCredentialCache } from "./bridge";

export interface StreamChatHandlers {
  /** Called once when the server confirms it's about to respond. */
  onStarted?: () => void;
  /** Called when the brain reaches for a tool. */
  onToolCall?: (tool: string) => void;
  /** Called with each word as it streams. */
  onChunk: (text: string) => void;
  /** Called once at the end with final metadata. */
  onDone?: (info: {
    turn: number;
    duration_ms: number;
    metadata: Record<string, unknown>;
  }) => void;
  /** Called on protocol or transport error. WS auto-closes after. */
  onError?: (msg: string) => void;
}

export interface StreamChatOptions {
  /** Sha-strings for any /upload-staged images attached to this turn. */
  imageShas?: string[];
  /** If this turn is an explicit reply to an outbound initiate, the
   *  audit row id (e.g. ``"ia_001"``). The bridge ingests it, records
   *  the ``replied_explicit`` state transition + memory re-render
   *  atomically with the chat turn, and surfaces the linked subject
   *  to the chat engine's system prompt. Foundation for v0.0.10's
   *  acknowledged_unclear detection. */
  replyToAuditId?: string;
  /** Time to establish the socket before surfacing an error. */
  openTimeoutMs?: number;
  /** Time after open to receive the first bridge frame. */
  firstFrameTimeoutMs?: number;
  /** Overall stream budget. */
  overallTimeoutMs?: number;
}

const DEFAULT_OPEN_TIMEOUT_MS = 10_000;
const DEFAULT_FIRST_FRAME_TIMEOUT_MS = 20_000;
const DEFAULT_OVERALL_TIMEOUT_MS = 120_000;

/**
 * Open a WS stream, send the message, and dispatch frames to the
 * provided handlers. Returns a `cancel()` function the caller can
 * invoke to abort the in-flight reply (closes the WS).
 */
export async function streamChat(
  persona: string,
  sessionId: string,
  message: string,
  handlers: StreamChatHandlers,
  options: StreamChatOptions = {},
): Promise<() => void> {
  let ws: WebSocket | null = null;
  let cancelled = false;
  let completed = false;
  let started = false;
  let retriedCredentials = false;
  let errorSent = false;
  let openTimer: ReturnType<typeof setTimeout> | null = null;
  let firstFrameTimer: ReturnType<typeof setTimeout> | null = null;
  let overallTimer: ReturnType<typeof setTimeout> | null = null;

  const openTimeoutMs = options.openTimeoutMs ?? DEFAULT_OPEN_TIMEOUT_MS;
  const firstFrameTimeoutMs = options.firstFrameTimeoutMs ?? DEFAULT_FIRST_FRAME_TIMEOUT_MS;
  const overallTimeoutMs = options.overallTimeoutMs ?? DEFAULT_OVERALL_TIMEOUT_MS;

  const clearTimer = (timer: ReturnType<typeof setTimeout> | null) => {
    if (timer !== null) clearTimeout(timer);
  };
  const clearTimers = () => {
    clearTimer(openTimer);
    clearTimer(firstFrameTimer);
    clearTimer(overallTimer);
    openTimer = null;
    firstFrameTimer = null;
    overallTimer = null;
  };

  const fail = (message: string) => {
    if (cancelled || completed || errorSent) return;
    errorSent = true;
    clearTimers();
    handlers.onError?.(message);
    ws?.close();
  };

  const connect = async () => {
    clearTimers();
    const creds = await getBridgeCredentials(persona);
    if (cancelled) return;

    const url = `ws://127.0.0.1:${creds.port}/stream/${sessionId}`;
    const protocols = creds.authToken ? ["bearer", creds.authToken] : undefined;
    ws = new WebSocket(url, protocols);
    started = false;
    completed = false;
    errorSent = false;

    openTimer = setTimeout(() => fail("websocket open timed out"), openTimeoutMs);
    overallTimer = setTimeout(() => fail("stream timed out"), overallTimeoutMs);

    ws.addEventListener("open", () => {
      if (cancelled || !ws) return;
      clearTimer(openTimer);
      openTimer = null;
      firstFrameTimer = setTimeout(
        () => fail("stream timed out waiting for first frame"),
        firstFrameTimeoutMs,
      );
      const frame: {
        message: string;
        image_shas?: string[];
        reply_to_audit_id?: string;
      } = { message };
      if (options.imageShas && options.imageShas.length > 0) frame.image_shas = options.imageShas;
      if (options.replyToAuditId) frame.reply_to_audit_id = options.replyToAuditId;
      ws.send(JSON.stringify(frame));
    });

    ws.addEventListener("message", (event) => {
      if (cancelled || completed) return;
      clearTimer(firstFrameTimer);
      firstFrameTimer = null;
      let frame: { type: string; [key: string]: unknown };
      try {
        frame = JSON.parse(event.data as string);
      } catch {
        fail("malformed frame");
        return;
      }
      switch (frame.type) {
        case "started":
          started = true;
          handlers.onStarted?.();
          break;
        case "tool_call":
          handlers.onToolCall?.((frame.tool as string) ?? "?");
          break;
        case "tool_result":
          break;
        case "reply_chunk":
          handlers.onChunk((frame.text as string) ?? "");
          break;
        case "done":
          completed = true;
          clearTimers();
          handlers.onDone?.({
            turn: (frame.turn as number) ?? 0,
            duration_ms: (frame.duration_ms as number) ?? 0,
            metadata: (frame.metadata as Record<string, unknown>) ?? {},
          });
          ws?.close();
          break;
        case "error":
          fail((frame.detail as string) ?? (frame.code as string) ?? "stream error");
          break;
        default:
          fail(`unknown stream frame: ${frame.type}`);
          break;
      }
    });

    ws.addEventListener("error", () => {
      if (cancelled || completed) return;
      if (!started && !retriedCredentials) return;
      fail("websocket error");
    });

    ws.addEventListener("close", (event) => {
      if (cancelled || completed || errorSent) return;
      const authishClose = event.code === 4001 || event.code === 1006;
      if (!started && authishClose && !retriedCredentials) {
        retriedCredentials = true;
        resetBridgeCredentialCache(persona);
        void connect().catch((e) => fail((e as Error).message));
        return;
      }
      fail(`ws closed before completion (${event.code || "unknown"}): ${event.reason || "no reason"}`);
    });
  };

  await connect();

  return () => {
    cancelled = true;
    clearTimers();
    ws?.close();
  };
}

