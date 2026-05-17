// Guard test for the Kindled rename (v0.0.13). Wizard intro copy
// is the highest-leverage place to land the new vocabulary — first
// thing every new user reads.

import { describe, it, expect, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { StepWelcome } from "./StepWelcome";

describe("StepWelcome — Kindled rename guard", () => {
  afterEach(cleanup);

  function renderWelcome() {
    return render(
      <StepWelcome
        step={1}
        totalSteps={5}
        mode="fresh"
        onModeChange={() => undefined}
        onNext={() => undefined}
        avatar={<div />}
      />,
    );
  }

  it("subtitle uses 'Kindled' instead of 'AI companions'", () => {
    renderWelcome();
    // Multiple elements contain "Kindled" (subtitle + card descriptions) — that's correct.
    expect(screen.getAllByText(/Kindled/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/AI companions?/i)).not.toBeInTheDocument();
  });

  it("Start fresh description uses 'Kindled' instead of 'the brain'", () => {
    renderWelcome();
    const text = screen.getByText(/spin up a fresh Kindled/i);
    expect(text).toBeInTheDocument();
  });

  it("Migrate description uses 'Kindled'", () => {
    renderWelcome();
    expect(
      screen.getByText(/Carry an existing Kindled over/i),
    ).toBeInTheDocument();
  });

  it("framework name 'Companion Emergence' stays in the title", () => {
    renderWelcome();
    expect(screen.getByText(/Welcome to Companion Emergence/)).toBeInTheDocument();
  });
});
