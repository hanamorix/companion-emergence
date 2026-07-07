// NotesToggle — standalone component tests.
// Mirrors PronounPicker.test.tsx / ModelPicker.test.tsx pattern.

import { describe, test, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { NotesToggle } from "./NotesToggle";
import * as bridge from "../../bridge";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("NotesToggle", () => {
  test("toggles notes on and shows the resolved folder", async () => {
    vi.spyOn(bridge, "setPersonaNotes").mockResolvedValue({
      ok: true,
      enabled: true,
      folder: "/Users/x/Documents/Nell Notes",
    });
    render(<NotesToggle persona="nell" enabled={false} folder={null} />);
    fireEvent.click(screen.getByRole("checkbox"));
    await waitFor(() => expect(screen.getByText(/Nell Notes/)).toBeInTheDocument());
    expect(bridge.setPersonaNotes).toHaveBeenCalledWith("nell", true);
  });

  test("re-syncs when the enabled prop arrives from a later state poll", () => {
    // Live report 2026-07-04: the toggle mounted before the poll delivered
    // notes_enabled=true and stayed unchecked forever. The checked state
    // must follow prop updates, not just the initial value.
    const { rerender } = render(<NotesToggle persona="nell" enabled={false} folder={null} />);
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    rerender(<NotesToggle persona="nell" enabled={true} folder="/Users/x/Documents/Nell Notes" />);
    expect(screen.getByRole("checkbox")).toBeChecked();
    expect(screen.getByText(/Nell Notes/)).toBeInTheDocument();
  });
});
