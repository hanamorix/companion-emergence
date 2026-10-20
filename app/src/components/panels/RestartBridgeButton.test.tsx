// Component tests for the bridge-restart button (#456 Q-A). The stop display has its own test (StopProgress.test.tsx).
//
// Hook integration is covered by useRestartBridge.test.ts; here the hook is mocked so each state can be driven.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("../../hooks/useRestartBridge");

import { RestartBridgeButton } from "./RestartBridgeButton";
import { useRestartBridge, type RestartState } from "../../hooks/useRestartBridge";

const mockedHook = vi.mocked(useRestartBridge);

function returnState(
  state: RestartState,
  errorDetail: string | null = null,
  fellThrough: string | null = null,
  extra: Partial<ReturnType<typeof useRestartBridge>> = {},
) {
  return {
    state,
    errorDetail,
    fellThrough,
    progress: null,
    offerHardExit: false,
    offerText: null,
    restart: vi.fn(),
    hardExit: vi.fn(),
    keepWaiting: vi.fn(),
    onModeChanged: vi.fn(),
    ...extra,
  };
}

beforeEach(() => {
  mockedHook.mockReset();
});

afterEach(() => {
  cleanup();
});

function restartButton() {
  return screen.getAllByRole("button")[0];
}

describe("RestartBridgeButton", () => {
  it("disabled during all non-idle, non-failed states", () => {
    const inFlight: RestartState[] = [
      "requesting",
      "stopping",
      "forcing",
      "starting",
      "waiting_for_health",
      "reconnecting",
      "success",
    ];
    for (const s of inFlight) {
      mockedHook.mockReturnValue(returnState(s));
      const { unmount } = render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
      expect(restartButton()).toBeDisabled();
      unmount();
    }
  });

  it("enabled in idle and failed states", () => {
    for (const s of ["idle", "failed"] as RestartState[]) {
      mockedHook.mockReturnValue(returnState(s));
      const { unmount } = render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
      expect(restartButton()).toBeEnabled();
      unmount();
    }
  });

  it("clicking the button in idle calls restart()", () => {
    const stub = returnState("idle");
    mockedHook.mockReturnValue(stub);
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    fireEvent.click(restartButton());
    expect(stub.restart).toHaveBeenCalledTimes(1);
  });

  it("button label changes per state and carries aria-live=polite", () => {
    mockedHook.mockReturnValue(returnState("stopping"));
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(restartButton()).toHaveTextContent(/waiting for the brain to finish/i);
    expect(restartButton()).toHaveAttribute("aria-live", "polite");
    cleanup();
    mockedHook.mockReturnValue(returnState("forcing"));
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(restartButton()).toHaveTextContent(/not answering/i);
  });

  it("failed state surfaces errorDetail below the button", () => {
    mockedHook.mockReturnValue(returnState("failed", "Restart failed. Try `nell service status`."));
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.getByText(/nell service status/i)).toBeInTheDocument();
    expect(restartButton()).toHaveTextContent(/retry/i);
  });

  it("(#365) says why the brain had to be ended", () => {
    mockedHook.mockReturnValue(
      returnState("forcing", null, "the bridge answered neither /health nor the stop request"),
    );
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.getByText(/had to end the brain/i)).toHaveTextContent(/answered neither/);
  });

  it("(#456 review S9) the button no longer carries the stop display (one floating display app-wide)", () => {
    mockedHook.mockReturnValue(
      returnState("stopping", null, null, { progress: "Stopping. Waiting for 1 task to finish: soul review." }),
    );
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.queryByTestId("stop-progress")).toBeNull();
  });
});
