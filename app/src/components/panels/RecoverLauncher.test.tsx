import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { RecoverLauncher } from "./RecoverLauncher";

vi.mock("../../bridge", () => ({
  invokeRecoverPreflight: vi.fn().mockResolvedValue({ mode: "source", missing: 2, unfade: 1 }),
  invokeRunRecover: vi
    .fn()
    .mockResolvedValue({ success: true, stdout: "", stderr: "", exit_code: 0 }),
}));

afterEach(cleanup);

describe("RecoverLauncher", () => {
  it("shows an entry button and reveals StepRecover preflight counts on click", async () => {
    render(<RecoverLauncher persona="Demo" />);
    // The entry affordance is visible to a single-persona user in the panel.
    const btn = screen.getByRole("button", { name: /recover memories/i });
    fireEvent.click(btn);
    // StepRecover loads preflight via the mocked bridge and renders the counts.
    expect(await screen.findByText(/2 memories to restore/i)).toBeInTheDocument();
    expect(await screen.findByText(/1 to un-fade/i)).toBeInTheDocument();
  });
});
