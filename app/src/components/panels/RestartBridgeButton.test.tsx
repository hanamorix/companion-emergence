// Component tests for the bridge-restart button and the stop display under it (#456 Q-A, D18, C17 e).
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

  it("(#456 D18) shows what is still running while the brain stops", () => {
    mockedHook.mockReturnValue(
      returnState("stopping", null, null, {
        progress: "Stopping. Waiting for 2 tasks to finish: chat reply, background job: compaction.",
      }),
    );
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.getByTestId("stop-progress")).toHaveTextContent(
      "Stopping. Waiting for 2 tasks to finish: chat reply, background job: compaction.",
    );
    expect(screen.queryByText("Hard exit")).toBeNull(); // no offer before 180 s
  });

  it("(#456 C17 e) the offer has Hard exit and Keep waiting, and no kill happens without one", () => {
    const stub = returnState("stopping", null, null, {
      progress: "Stopping. Waiting for 1 task to finish: soul review.",
      offerHardExit: true,
      offerText:
        "Still waiting for 1 task after 3 minutes: soul review. Hard exit now? The next start will pick up anything left unfinished.",
    });
    mockedHook.mockReturnValue(stub);
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.getByText(/Hard exit now\?/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Keep waiting"));
    expect(stub.keepWaiting).toHaveBeenCalledTimes(1);
    expect(stub.hardExit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Hard exit"));
    expect(stub.hardExit).toHaveBeenCalledTimes(1);
  });
});
