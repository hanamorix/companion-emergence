// The stop display and the hard-exit offer (#456 Q-A, D18, C17 e; review S9). The hook is mocked so each state can be
// driven; the hook itself is covered by useRestartBridge.test.ts.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("../../hooks/useRestartBridge");

import { StopProgress } from "./StopProgress";
import { useRestartBridge, type RestartState } from "../../hooks/useRestartBridge";

const mockedHook = vi.mocked(useRestartBridge);

function returnState(state: RestartState, extra: Partial<ReturnType<typeof useRestartBridge>> = {}) {
  return {
    state,
    errorDetail: null,
    fellThrough: null,
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

describe("StopProgress", () => {
  it("renders nothing when no stop is being watched", () => {
    mockedHook.mockReturnValue(returnState("idle"));
    render(<StopProgress persona="p" currentMode="live" floating />);
    expect(screen.queryByTestId("stop-progress")).toBeNull();
  });

  it("(#456 D18) shows what is still running while the brain stops", () => {
    mockedHook.mockReturnValue(
      returnState("stopping", {
        progress: "Stopping. Waiting for 2 tasks to finish: chat reply, background job: compaction.",
      }),
    );
    render(<StopProgress persona="p" currentMode="bridge_down" />);
    expect(screen.getByTestId("stop-progress")).toHaveTextContent(
      "Stopping. Waiting for 2 tasks to finish: chat reply, background job: compaction.",
    );
    expect(screen.queryByText("Hard exit")).toBeNull(); // no offer before 180 s
  });

  it("(#456 C17 e) the offer has Hard exit and Keep waiting, and no kill happens without one", () => {
    const stub = returnState("stopping", {
      progress: "Stopping. Waiting for 1 task to finish: soul review.",
      offerHardExit: true,
      offerText:
        "Still waiting for 1 task after 3 minutes: soul review. Hard exit now? The next start will pick up anything left unfinished.",
    });
    mockedHook.mockReturnValue(stub);
    render(<StopProgress persona="p" currentMode="bridge_down" />);
    expect(screen.getByText(/Hard exit now\?/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Keep waiting"));
    expect(stub.keepWaiting).toHaveBeenCalledTimes(1);
    expect(stub.hardExit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Hard exit"));
    expect(stub.hardExit).toHaveBeenCalledTimes(1);
  });

  it("(review S9) the floating display sits over the main screen", () => {
    mockedHook.mockReturnValue(returnState("stopping", { progress: "Stopping." }));
    render(<StopProgress persona="p" currentMode="live" floating />);
    expect(screen.getByTestId("stop-progress")).toHaveStyle({ position: "fixed" });
  });
});
