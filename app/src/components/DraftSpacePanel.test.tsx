import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { DraftSpacePanel } from "./DraftSpacePanel";

afterEach(() => {
  cleanup();
});

describe("DraftSpacePanel", () => {
  it("renders markdown content from the draft file", () => {
    const markdown = "## 2026-05-11 14:32 (dream)\n\nThe dream wasn't loud enough.";
    render(<DraftSpacePanel markdown={markdown} persona="nell" />);
    expect(screen.getByText(/wasn't loud enough/)).toBeInTheDocument();
  });

  it("shows an empty-state message when markdown is empty", () => {
    render(<DraftSpacePanel markdown="" persona="nell" />);
    expect(screen.getByText(/no drafts yet/i)).toBeInTheDocument();
  });

  it("uses companion name in heading, not hardcoded 'Nell'", () => {
    render(<DraftSpacePanel markdown="some content" persona="mira" />);
    expect(screen.getByRole("heading")).toHaveTextContent(/Mira/);
    expect(screen.getByRole("heading").textContent).not.toContain("Nell");
  });
});
