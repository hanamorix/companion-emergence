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
import { friendlyChatError } from "./lib/friendlyChatError";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  listeners = new Map<string, Array<(event: { data?: string }) => void>>();
  closed = false;

  constructor(_url: string, _protocols?: string[]) {
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, handler: (event: { data?: string }) => void) {
    const list = this.listeners.get(type) ?? [];
    list.push(handler);
    this.listeners.set(type, list);
  }

  send(_data: string) {}

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

async function errorTextFor(frame: Record<string, unknown>): Promise<string> {
  const onError = vi.fn();
  await streamChat("alice", "s1", "hi", { onChunk: vi.fn(), onError });
  await flushMicrotasks();
  const ws = FakeWebSocket.instances[0];
  ws.emit("open");
  ws.emit("message", { data: JSON.stringify({ type: "started", session_id: "s1", at: "t" }) });
  ws.emit("message", { data: JSON.stringify({ type: "error", done: true, ...frame }) });
  expect(onError).toHaveBeenCalledTimes(1);
  return onError.mock.calls[0]![0] as string;
}

describe("streamChat error frames for a rejected model (#440)", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("passes the model on so the message can name it", async () => {
    const text = await errorTextFor({ code: "model_unavailable", model: "claude-sonet-5" });
    expect(text).toBe("model_unavailable:claude-sonet-5");
    expect(friendlyChatError(text)).toContain('"claude-sonet-5"');
  });

  it("falls back to the bare code when the frame carries no model", async () => {
    expect(await errorTextFor({ code: "model_unavailable" })).toBe("model_unavailable");
  });

  it("leaves every other error frame as before", async () => {
    expect(await errorTextFor({ code: "provider_failed" })).toBe("provider_failed");
  });

  it("still prefers detail for other codes", async () => {
    expect(await errorTextFor({ code: "x", detail: "because" })).toBe("because");
  });
});
