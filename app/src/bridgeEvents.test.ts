import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./bridge", () => ({
  getBridgeCredentials: vi.fn(async () => ({
    url: "http://127.0.0.1:41001",
    port: 41001,
    authToken: "tok",
  })),
}));

import { getBridgeCredentials } from "./bridge";
import { subscribeToBridgeEvents } from "./bridgeEvents";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  protocols?: string[];
  listeners = new Map<string, Array<(event: { data?: string }) => void>>();

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

  close() {
    this.emit("close");
  }

  emit(type: string, event: { data?: string } = {}) {
    for (const handler of this.listeners.get(type) ?? []) handler(event);
  }
}

describe("subscribeToBridgeEvents reconnect", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  async function flushMicrotasks() {
    await Promise.resolve();
    await Promise.resolve();
  }

  it("reconnects /events with bounded backoff after close", async () => {
    const stream = subscribeToBridgeEvents("alice");
    await flushMicrotasks();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).toBe("ws://127.0.0.1:41001/events");
    expect(FakeWebSocket.instances[0].protocols).toEqual(["bearer", "tok"]);

    FakeWebSocket.instances[0].emit("close");
    await vi.advanceTimersByTimeAsync(499);
    expect(FakeWebSocket.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await flushMicrotasks();
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.instances[1].emit("close");
    await vi.advanceTimersByTimeAsync(1000);
    await flushMicrotasks();
    expect(FakeWebSocket.instances).toHaveLength(3);

    stream.close();
  });

  it("caps reconnect delay at 10 seconds", async () => {
    subscribeToBridgeEvents("alice");
    await flushMicrotasks();

    const delays = [500, 1000, 2000, 5000, 10000, 10000];
    for (const delay of delays) {
      FakeWebSocket.instances.at(-1)?.emit("close");
      await vi.advanceTimersByTimeAsync(delay - 1);
      const before = FakeWebSocket.instances.length;
      await vi.advanceTimersByTimeAsync(1);
      await flushMicrotasks();
      expect(FakeWebSocket.instances.length).toBe(before + 1);
    }
    expect(getBridgeCredentials).toHaveBeenCalledTimes(delays.length + 1);
  });

  it("does not reconnect after close() cleanup", async () => {
    const stream = subscribeToBridgeEvents("alice");
    await flushMicrotasks();
    stream.close();
    FakeWebSocket.instances[0].emit("close");
    await vi.advanceTimersByTimeAsync(10_000);
    await flushMicrotasks();
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});
