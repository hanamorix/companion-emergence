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
});
