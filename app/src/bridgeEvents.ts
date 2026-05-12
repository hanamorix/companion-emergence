/**
 * Bridge /events WebSocket subscription.
 *
 * The bridge exposes a server-push only WebSocket at ``/events`` that
 * broadcasts physiology + initiate events to any connected renderer.
 * This module wraps the raw socket in a minimal pub-sub abstraction
 * so callers (currently ChatPanel) can subscribe to typed events
 * without managing the socket lifecycle themselves.
 *
 * The connection is best-effort: if the bridge isn't reachable, we
 * fail quietly — ambient banners are a nice-to-have, not a critical
 * path. The caller's component still renders.
 */
import { getBridgeCredentials } from "./bridge";

export type BridgeEvent = Record<string, unknown> & { type: string };

export interface EventStream {
  /** Register a handler; returns an unsubscribe function. */
  subscribe(handler: (event: BridgeEvent) => void): () => void;
}

/**
 * Open a /events WebSocket scoped to ``persona`` and return an
 * EventStream the caller can subscribe to. Calling unsubscribe on the
 * stream (or the returned cleanup) closes the socket.
 *
 * If the bridge is unreachable the stream is still returned — handlers
 * just never fire. This lets ChatPanel unconditionally subscribe in
 * useEffect without needing to special-case the bridge-down state.
 */
export function subscribeToBridgeEvents(persona: string): EventStream & { close: () => void } {
  const handlers = new Set<(event: BridgeEvent) => void>();
  let ws: WebSocket | null = null;
  let closed = false;

  void (async () => {
    try {
      const creds = await getBridgeCredentials(persona);
      if (closed) return;
      const url = `ws://127.0.0.1:${creds.port}/events`;
      const protocols = creds.authToken ? ["bearer", creds.authToken] : undefined;
      ws = new WebSocket(url, protocols);
      ws.addEventListener("message", (e) => {
        try {
          const data = JSON.parse(e.data as string) as BridgeEvent;
          for (const h of handlers) h(data);
        } catch {
          // Ignore malformed frames — server-side concern.
        }
      });
    } catch {
      // Bridge unreachable. The stream stays inert.
    }
  })();

  return {
    subscribe(handler) {
      handlers.add(handler);
      return () => handlers.delete(handler);
    },
    close() {
      closed = true;
      handlers.clear();
      try {
        ws?.close();
      } catch {
        // Already closed.
      }
      ws = null;
    },
  };
}
