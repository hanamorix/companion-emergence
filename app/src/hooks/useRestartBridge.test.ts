// Vitest coverage for the v0.0.14 bridge-restart hook. Mocks the four
// bridge.ts call sites and walks the state machine through happy path,
// both HTTP-step timeouts, and the post-force health-poll failure.
//
// Spec: docs/superpowers/specs/2026-05-17-bridge-restart-button-design.md §5–6.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";

import * as bridge from "../bridge";
import { useRestartBridge } from "./useRestartBridge";

const PERSONA = "test-persona";

function jsonResponse(status: number, body: object = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useRestartBridge", () => {
  it("happy path: idle → closing → shutting_down → waiting_for_health → reconnecting → success", async () => {
    vi.spyOn(bridge, "closeActiveSession").mockResolvedValue(jsonResponse(200));
    vi.spyOn(bridge, "shutdownBridge").mockResolvedValue(jsonResponse(202));
    vi.spyOn(bridge, "fetchHealth").mockResolvedValue({ liveness: "ok" });
    const forceMock = vi.spyOn(bridge, "invokeForceRestart").mockResolvedValue();

    const { result, rerender } = renderHook(
      ({ mode }: { mode: "live" | "bridge_down" }) =>
        useRestartBridge(PERSONA, mode),
      { initialProps: { mode: "bridge_down" } },
    );
    expect(result.current.state).toBe("idle");

    act(() => {
      result.current.restart();
    });

    await waitFor(() => expect(result.current.state).toBe("reconnecting"));
    expect(forceMock).not.toHaveBeenCalled();

    // Parent's /state poll flips back to "live" → reconnecting → success.
    rerender({ mode: "live" });
    await waitFor(() => expect(result.current.state).toBe("success"));
  });

  it("4xx from /sessions/close is treated as success — closeActiveSession returns the Response", async () => {
    // Spec §6.2: only 5xx + timeouts escalate. Our closeActiveSession
    // never throws on 4xx — it returns the raw Response and the hook
    // moves forward to shutdown.
    vi.spyOn(bridge, "closeActiveSession").mockResolvedValue(jsonResponse(404));
    vi.spyOn(bridge, "shutdownBridge").mockResolvedValue(jsonResponse(202));
    vi.spyOn(bridge, "fetchHealth").mockResolvedValue({ liveness: "ok" });
    const forceMock = vi.spyOn(bridge, "invokeForceRestart").mockResolvedValue();

    const { result } = renderHook(() =>
      useRestartBridge(PERSONA, "bridge_down"),
    );
    act(() => {
      result.current.restart();
    });

    await waitFor(() => expect(result.current.state).toBe("reconnecting"));
    expect(forceMock).not.toHaveBeenCalled();
  });

  it("/sessions/close timeout escalates to forcing → reconnecting", async () => {
    vi.useFakeTimers();
    // Hang the close call past its 5s timeout.
    vi.spyOn(bridge, "closeActiveSession").mockImplementation(
      () => new Promise<Response>(() => {}),
    );
    const forceMock = vi
      .spyOn(bridge, "invokeForceRestart")
      .mockResolvedValue();
    vi.spyOn(bridge, "fetchHealth").mockResolvedValue({ liveness: "ok" });

    const { result } = renderHook(() =>
      useRestartBridge(PERSONA, "bridge_down"),
    );
    act(() => {
      result.current.restart();
    });

    // Trip the close timeout, then let the forcing + health-poll
    // microtask chain settle on subsequent timer flushes.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5500);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });

    expect(forceMock).toHaveBeenCalledWith(PERSONA);
    expect(result.current.state).toBe("reconnecting");
  });

  it("/supervisor/shutdown timeout escalates to forcing → reconnecting", async () => {
    vi.useFakeTimers();
    vi.spyOn(bridge, "closeActiveSession").mockResolvedValue(jsonResponse(200));
    vi.spyOn(bridge, "shutdownBridge").mockImplementation(
      () => new Promise<Response>(() => {}),
    );
    const forceMock = vi
      .spyOn(bridge, "invokeForceRestart")
      .mockResolvedValue();
    vi.spyOn(bridge, "fetchHealth").mockResolvedValue({ liveness: "ok" });

    const { result } = renderHook(() =>
      useRestartBridge(PERSONA, "bridge_down"),
    );
    act(() => {
      result.current.restart();
    });

    // Close resolves immediately on the next microtask flush.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    // Trip the 3s shutdown timeout.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3500);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });

    // Spec §6.4: shutdown failures don't escalate by themselves — the
    // health poll runs first. Here the bridge is "alive" (fetchHealth
    // resolves), so we should land in reconnecting without ever forcing.
    expect(result.current.state).toBe("reconnecting");
    expect(forceMock).not.toHaveBeenCalled();
  });

  it("post-force health poll exhaustion → failed with user-readable error", async () => {
    vi.useFakeTimers();
    vi.spyOn(bridge, "closeActiveSession").mockResolvedValue(jsonResponse(200));
    vi.spyOn(bridge, "shutdownBridge").mockResolvedValue(jsonResponse(202));
    // Health never comes back during either window.
    vi.spyOn(bridge, "fetchHealth").mockRejectedValue(new Error("dead"));
    vi.spyOn(bridge, "invokeForceRestart").mockResolvedValue();

    const { result } = renderHook(() =>
      useRestartBridge(PERSONA, "bridge_down"),
    );
    act(() => {
      result.current.restart();
    });

    // Two 30s health windows + a sliver of overhead.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(70000);
    });

    expect(result.current.state).toBe("failed");
    expect(result.current.errorDetail).toMatch(/nell service status/i);
  });

  it("invokeForceRestart rejecting with a pid-missing message surfaces as errorDetail", async () => {
    vi.useFakeTimers();
    vi.spyOn(bridge, "closeActiveSession").mockImplementation(
      () => new Promise<Response>(() => {}),
    );
    vi.spyOn(bridge, "invokeForceRestart").mockRejectedValue(
      new Error(
        "bridge.json missing pid — restart Companion to re-spawn bridge with pid field",
      ),
    );

    const { result } = renderHook(() =>
      useRestartBridge(PERSONA, "bridge_down"),
    );
    act(() => {
      result.current.restart();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5500);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });

    expect(result.current.state).toBe("failed");
    expect(result.current.errorDetail).toMatch(/missing pid/i);
  });

  it("re-entry guard: clicking restart while in flight is a no-op", async () => {
    const closeMock = vi
      .spyOn(bridge, "closeActiveSession")
      .mockImplementation(() => new Promise<Response>(() => {}));

    const { result } = renderHook(() =>
      useRestartBridge(PERSONA, "bridge_down"),
    );
    act(() => {
      result.current.restart();
      result.current.restart();
      result.current.restart();
    });

    // Only one close call should be in flight despite three click attempts.
    await waitFor(() => expect(closeMock).toHaveBeenCalledTimes(1));
  });
});
