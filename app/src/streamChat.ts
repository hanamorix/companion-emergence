/**
 * WS streaming chat client — talks to /stream/{session_id}.
 *
 * Wire protocol (from brain/bridge/server.py):
 *   client → server: {message: string} (one-shot, after WS open)
 *   server → client (sequence):
 *     {type: "started", session_id, at}
 *     {type: "tool_call", tool, session_id, at}     × per tool
 *     {type: "tool_result", tool, summary, ...}     × per tool
 *     {type: "reply_chunk", text}                   × per word
 *     {type: "done", session_id, turn, duration_ms, metadata, at}
 *   on error:
 *     {type: "error", code, detail?, done: true}
 *
 * Auth: Sec-WebSocket-Protocol: bearer, <token>. Same shape as the CLI's
 * `nell bridge tail` after the audit-2 I-1 fix.
 */

import { getBridgeCredentials } from "./bridge";

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
}

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
  const creds = await getBridgeCredentials(persona);
  const url = `ws://127.0.0.1:${creds.port}/stream/${sessionId}`;
  const protocols = creds.authToken ? ["bearer", creds.authToken] : undefined;

  const ws = new WebSocket(url, protocols);
  let cancelled = false;

  ws.addEventListener("open", () => {
    if (cancelled) return;
    const frame: { message: string; image_shas?: string[] } = { message };
    if (options.imageShas && options.imageShas.length > 0) {
      frame.image_shas = options.imageShas;
    }
    ws.send(JSON.stringify(frame));
  });

  ws.addEventListener("message", (event) => {
    if (cancelled) return;
    let frame: { type: string; [key: string]: unknown };
    try {
      frame = JSON.parse(event.data as string);
    } catch {
      handlers.onError?.("malformed frame");
      ws.close();
      return;
    }
    switch (frame.type) {
      case "started":
        handlers.onStarted?.();
        break;
      case "tool_call":
        handlers.onToolCall?.((frame.tool as string) ?? "?");
        break;
      case "tool_result":
        // tool result events arrive between tool_call and chunks — UI
        // doesn't surface them in v1; bridge audit log is canonical.
        break;
      case "reply_chunk":
        handlers.onChunk((frame.text as string) ?? "");
        break;
      case "done":
        handlers.onDone?.({
          turn: (frame.turn as number) ?? 0,
          duration_ms: (frame.duration_ms as number) ?? 0,
          metadata: (frame.metadata as Record<string, unknown>) ?? {},
        });
        ws.close();
        break;
      case "error":
        handlers.onError?.(
          (frame.detail as string) ?? (frame.code as string) ?? "stream error",
        );
        ws.close();
        break;
    }
  });

  ws.addEventListener("error", () => {
    if (cancelled) return;
    handlers.onError?.("websocket error");
  });

  ws.addEventListener("close", (event) => {
    if (cancelled) return;
    if (event.code !== 1000 && event.code !== 1005) {
      handlers.onError?.(`ws closed (${event.code}): ${event.reason || "unknown"}`);
    }
  });

  return () => {
    cancelled = true;
    ws.close();
  };
}
