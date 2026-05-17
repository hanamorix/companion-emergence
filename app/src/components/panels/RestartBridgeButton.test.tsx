// Component tests for the v0.0.14 bridge-restart button. Verifies the
// disabled-during-progress invariant, label-per-state, and the
// aria-live=polite attribute the spec promised screen-reader users.
//
// Hook integration is covered by useRestartBridge.test.ts — here we
// mock the hook so we can drive each state independently.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("../../hooks/useRestartBridge");

import { RestartBridgeButton } from "./RestartBridgeButton";
import {
  useRestartBridge,
  type RestartState,
} from "../../hooks/useRestartBridge";

const mockedHook = vi.mocked(useRestartBridge);

function returnState(state: RestartState, errorDetail: string | null = null) {
  return {
    state,
    errorDetail,
    restart: vi.fn(),
    onModeChanged: vi.fn(),
  };
}

beforeEach(() => {
  mockedHook.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("RestartBridgeButton", () => {
  it("disabled during all non-idle, non-failed states", () => {
    const inFlight: RestartState[] = [
      "closing",
      "shutting_down",
      "waiting_for_health",
      "forcing",
      "reconnecting",
      "success",
    ];
    for (const s of inFlight) {
      mockedHook.mockReturnValue(returnState(s));
      const { unmount } = render(
        <RestartBridgeButton persona="p" currentMode="bridge_down" />,
      );
      expect(screen.getByRole("button")).toBeDisabled();
      unmount();
    }
  });

  it("enabled in idle and failed states", () => {
    for (const s of ["idle", "failed"] as RestartState[]) {
      mockedHook.mockReturnValue(returnState(s));
      const { unmount } = render(
        <RestartBridgeButton persona="p" currentMode="bridge_down" />,
      );
      expect(screen.getByRole("button")).toBeEnabled();
      unmount();
    }
  });

  it("clicking the button in idle calls restart()", () => {
    const stub = returnState("idle");
    mockedHook.mockReturnValue(stub);
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    fireEvent.click(screen.getByRole("button"));
    expect(stub.restart).toHaveBeenCalledTimes(1);
  });

  it("button label changes per state and carries aria-live=polite", () => {
    mockedHook.mockReturnValue(returnState("closing"));
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    const btn = screen.getByRole("button");
    expect(btn).toHaveTextContent(/ending conversation/i);
    expect(btn).toHaveAttribute("aria-live", "polite");

    cleanup();
    mockedHook.mockReturnValue(returnState("forcing"));
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.getByRole("button")).toHaveTextContent(
      /bridge not responding/i,
    );
  });

  it("failed state surfaces errorDetail below the button", () => {
    mockedHook.mockReturnValue(
      returnState("failed", "Restart failed. Try `nell service status`."),
    );
    render(<RestartBridgeButton persona="p" currentMode="bridge_down" />);
    expect(screen.getByText(/nell service status/i)).toBeInTheDocument();
    // And the button reverts to a retry affordance.
    expect(screen.getByRole("button")).toHaveTextContent(/retry/i);
  });
});
