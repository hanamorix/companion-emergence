import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./bridge", () => ({
  getBridgeCredentials: vi.fn(async () => ({
    url: "http://127.0.0.1:41001",
    port: 41001,
    authToken: "tok",
  })),
  resetBridgeCredentialCache: vi.fn(),
}));

import { streamChat } from "./streamChat";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  protocols?: string[];
  listeners = new Map<string, Array<(event: { data?: string }) => void>>();
  sent: string[] = [];
  closed = false;

  constructor(url: string, protocols?: string[]) {
    this.url = url;
    this.protocols = protocols;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, handler: (event: { data?: string }) => void) {
    const list = this.listeners.get(type) ?? [];
    list.push(handler);
    this.listeners.set(type, list);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.closed = true;
    this.emit("close", {});
  }

  emit(type: string, event: { data?: string } = {}) {
    for (const handler of this.listeners.get(type) ?? []) handler(event);
  }
}

async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
}

describe("streamChat keepalive frame", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("treats a keepalive frame as a benign no-op (no error, no chunk, stream stays open)", async () => {
    const onChunk = vi.fn();
    const onError = vi.fn();
    await streamChat("alice", "s1", "hi", { onChunk, onError });
    await flushMicrotasks();

    const ws = FakeWebSocket.instances[0];
    ws.emit("open");
    ws.emit("message", {
      data: JSON.stringify({ type: "started", session_id: "s1", at: "t" }),
    });
    // The server emits this during a silent provider stretch (first-token
    // latency / tool round-trip) to hold the WS open. It must NOT fail the
    // stream and must NOT be rendered as reply text.
    ws.emit("message", { data: JSON.stringify({ type: "keepalive", at: "t" }) });

    expect(onError).not.toHaveBeenCalled();
    expect(onChunk).not.toHaveBeenCalled();
    expect(ws.closed).toBe(false);
  });
});
