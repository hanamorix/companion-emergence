import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { StepReview } from "./StepReview";
import type { WizardState } from "../Wizard";

function renderReview(chatModel: string) {
  const state: WizardState = {
    mode: "fresh",
    personaName: "test",
    userName: "",
    userPronouns: "they/them",
    voiceTemplate: "default",
    chatModel,
    migrateFromPath: "",
    migrateSource: "nellbrain",
  };
  render(
    <StepReview
      step={6}
      totalSteps={7}
      state={state}
      preflight={null}
      onInstall={vi.fn()}
      onBack={vi.fn()}
      avatar={null}
    />,
  );
}

describe("StepReview Claude model row (#530 review)", () => {
  afterEach(cleanup);

  it("says a blank model keeps the install's current setting, not the default", () => {
    renderReview("");
    expect(screen.getByText("(keep the current setting)")).toBeInTheDocument();
    expect(screen.queryByText(/keep the default/)).not.toBeInTheDocument();
  });

  it("shows the chosen model when one is set", () => {
    renderReview("opus");
    expect(screen.getByText("opus")).toBeInTheDocument();
  });
});
