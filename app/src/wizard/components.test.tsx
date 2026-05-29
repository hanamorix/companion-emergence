import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import "@testing-library/jest-dom/vitest";
import { WizardShell } from "./components";

describe("WizardShell", () => {
  it("renders title without step counter when step/totalSteps are absent", () => {
    render(
      <WizardShell title="Recover memories">
        <div>content</div>
      </WizardShell>
    );
    expect(screen.getByText("Recover memories")).toBeInTheDocument();
    expect(screen.queryByText(/step \d+ of \d+/i)).not.toBeInTheDocument();
  });

  it("renders step counter when step and totalSteps are provided", () => {
    render(
      <WizardShell title="Name" step={2} totalSteps={5}>
        <div>content</div>
      </WizardShell>
    );
    expect(screen.getByText(/step 2 of 5/i)).toBeInTheDocument();
  });
});
