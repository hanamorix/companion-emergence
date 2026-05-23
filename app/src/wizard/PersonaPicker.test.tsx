import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// WizardAvatar pulls in import.meta.glob expressions — stub it out.
vi.mock("./Avatar", () => ({
  WizardAvatar: () => null,
}));

import { PersonaPicker } from "./PersonaPicker";
import type { PersonaSummary } from "../appConfig";

const personas: PersonaSummary[] = [
  { name: "nell",   last_opened_at: "2026-05-22T10:00:00Z", has_memories_db: true },
  { name: "phoebe", last_opened_at: "2026-05-23T09:00:00Z", has_memories_db: true },
  { name: "broken", last_opened_at: null,                   has_memories_db: false },
];

describe("PersonaPicker", () => {
  afterEach(cleanup);

  it("lists all personas", () => {
    render(<PersonaPicker personas={personas} onPick={() => {}} onNew={() => {}} />);
    ["nell", "phoebe", "broken"].forEach((n) =>
      expect(screen.getByText(new RegExp(n))).toBeInTheDocument()
    );
  });

  it("calls onPick with the chosen name", () => {
    const onPick = vi.fn();
    render(<PersonaPicker personas={personas} onPick={onPick} onNew={() => {}} />);
    fireEvent.click(screen.getByText(/phoebe/));
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onPick).toHaveBeenCalledWith("phoebe");
  });

  it("calls onNew when 'set up a new one' clicked", () => {
    const onNew = vi.fn();
    render(<PersonaPicker personas={personas} onPick={() => {}} onNew={onNew} />);
    fireEvent.click(screen.getByText(/set up a new one/i));
    expect(onNew).toHaveBeenCalled();
  });

  it("flags personas without memories.db as incomplete", () => {
    render(<PersonaPicker personas={personas} onPick={() => {}} onNew={() => {}} />);
    expect(screen.getByText(/incomplete/i)).toBeInTheDocument();
  });
});
