import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { InnerWeatherPanel } from "./InnerWeatherPanel";

// Minimal PersonaState shape — only the fields InnerWeatherPanel reads.
function fakeState(emotions: Record<string, number> | null, body: Record<string, unknown> | null = null) {
  return { emotions, body } as any;
}

describe("InnerWeatherPanel", () => {
  afterEach(() => cleanup());

  it("renders up to 7 emotion bars (raised from the old 5-bar cutoff)", () => {
    const emotions = {
      a: 1,
      b: 2,
      c: 3,
      d: 4,
      e: 5,
      f: 6,
      g: 7,
      h: 8, // 8th entry — must NOT render, only top 7
    };
    render(<InnerWeatherPanel state={fakeState(emotions)} />);
    for (const name of ["a", "b", "c", "d", "e", "f", "g"]) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
    expect(screen.queryByText("h")).not.toBeInTheDocument();
  });

  it("shows no delta arrow on the first render (no prior poll to compare)", () => {
    render(<InnerWeatherPanel state={fakeState({ joy: 5 })} />);
    expect(screen.queryByText("▲")).not.toBeInTheDocument();
    expect(screen.queryByText("▼")).not.toBeInTheDocument();
  });

  it("shows an up arrow when an emotion rises by more than 0.2 between polls", () => {
    const { rerender } = render(<InnerWeatherPanel state={fakeState({ joy: 5 })} />);
    rerender(<InnerWeatherPanel state={fakeState({ joy: 5.5 })} />);
    expect(screen.getByText("▲")).toBeInTheDocument();
  });

  it("shows a down arrow when an emotion falls by more than 0.2 between polls", () => {
    const { rerender } = render(<InnerWeatherPanel state={fakeState({ joy: 5 })} />);
    rerender(<InnerWeatherPanel state={fakeState({ joy: 4.5 })} />);
    expect(screen.getByText("▼")).toBeInTheDocument();
  });

  it("shows no arrow when the delta is within the 0.2 threshold", () => {
    const { rerender } = render(<InnerWeatherPanel state={fakeState({ joy: 5 })} />);
    rerender(<InnerWeatherPanel state={fakeState({ joy: 5.1 })} />);
    expect(screen.queryByText("▲")).not.toBeInTheDocument();
    expect(screen.queryByText("▼")).not.toBeInTheDocument();
  });
});
