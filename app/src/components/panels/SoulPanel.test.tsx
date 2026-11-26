import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import type { SoulHighlight } from "../../bridge";
import { SoulPanel } from "./SoulPanel";

// Minimal PersonaState shape — only the soul fields SoulPanel reads.
function fakeState(soul: Record<string, unknown>) {
  return soul as any;
}

function moment(text: string, resonance: number, when: string): SoulHighlight {
  return {
    id: `id-${text}`,
    moment: text,
    love_type: "belonging",
    resonance,
    crystallized_at: when,
    why_it_matters: null,
  };
}

const NEWEST = moment("a newer resonance-7 moment", 7, "2026-11-22T12:00:00+00:00");
const STRONGEST = moment("the old resonance-10 moment", 10, "2026-09-10T12:00:00+00:00");

describe("SoulPanel (#637)", () => {
  afterEach(() => cleanup());

  it("shows the newest moment as the card and the strongest one beside it", () => {
    render(<SoulPanel state={fakeState({ soul_highlight: NEWEST, soul_strongest: STRONGEST })} />);

    expect(screen.getByText(/a newer resonance-7 moment/)).toBeInTheDocument();
    const strongest = screen.getByTestId("soul-strongest");
    expect(strongest).toHaveTextContent("Strongest");
    expect(strongest).toHaveTextContent("the old resonance-10 moment");
    expect(strongest).toHaveTextContent("Resonance 10");
  });

  it("shows no strongest line when the newest moment is also the strongest, or an older brain sends none", () => {
    for (const soul of [{ soul_highlight: NEWEST, soul_strongest: null }, { soul_highlight: NEWEST }]) {
      render(<SoulPanel state={fakeState(soul)} />);
      expect(screen.getByText(/a newer resonance-7 moment/)).toBeInTheDocument();
      expect(screen.queryByTestId("soul-strongest")).not.toBeInTheDocument();
      cleanup();
    }
  });
});
