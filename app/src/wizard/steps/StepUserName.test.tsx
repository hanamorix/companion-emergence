import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { StepUserName } from "./StepUserName";

describe("StepUserName — pronoun pills", () => {
  afterEach(cleanup);

  it("renders three pronoun pills with the passed selection pressed", () => {
    render(
      <StepUserName
        step={5}
        totalSteps={9}
        value=""
        onChange={vi.fn()}
        onNext={vi.fn()}
        onBack={vi.fn()}
        avatar={null}
        pronouns="they/them"
        onPronounsChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "she/her" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "he/him" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "they/them" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("clicking a pill calls onPronounsChange with that preset", () => {
    const onChange = vi.fn();
    render(
      <StepUserName
        step={5}
        totalSteps={9}
        value=""
        onChange={vi.fn()}
        onNext={vi.fn()}
        onBack={vi.fn()}
        avatar={null}
        pronouns="they/them"
        onPronounsChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "he/him" }));
    expect(onChange).toHaveBeenCalledWith("he/him");
  });
});
