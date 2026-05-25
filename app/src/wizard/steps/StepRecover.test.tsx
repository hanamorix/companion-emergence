import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import "@testing-library/jest-dom/vitest";
import { StepRecover } from "./StepRecover";

vi.mock("../../bridge", () => ({
  invokeRecoverPreflight: vi.fn().mockResolvedValue({ mode: "source", missing: 3, unfade: 1 }),
  invokeRunRecover: vi.fn().mockResolvedValue({ success: true, stdout: "", stderr: "", exit_code: 0 }),
}));

describe("StepRecover", () => {
  it("renders preflight counts once loaded", async () => {
    render(<StepRecover persona="Phoebe" sourceDir="/src" onDone={() => {}} />);
    expect(await screen.findByText(/3 memories to restore/i)).toBeInTheDocument();
    expect(await screen.findByText(/1 to un-fade/i)).toBeInTheDocument();
  });
});
