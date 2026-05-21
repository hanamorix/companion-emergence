// ModelPicker — standalone component tests.
// Verifies render, apply action, and cost hint copy.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd === "get_bridge_credentials") {
      return { port: 50000, auth_token: null };
    }
    throw new Error(`unexpected cmd ${cmd}`);
  }),
}));

import { ModelPicker } from "./ModelPicker";

describe("ModelPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders all three model options with cost hints", () => {
    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );

    expect(screen.getByLabelText(/sonnet/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/opus/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/haiku/i)).toBeInTheDocument();

    // Cost hints are visible
    expect(screen.getByText(/recommended/i)).toBeInTheDocument();
    expect(screen.getByText(/smartest/i)).toBeInTheDocument();
    expect(screen.getByText(/fastest.*cheapest/i)).toBeInTheDocument();
  });

  it("Apply button is disabled when the selection matches current", () => {
    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).toBeDisabled();
  });

  it("Apply button becomes enabled after selecting a different model", () => {
    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );
    fireEvent.click(screen.getByLabelText(/opus/i));
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).not.toBeDisabled();
  });

  it("calls onClose with the new model after a successful apply", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }),
      ),
    );

    const onClose = vi.fn();
    render(
      <ModelPicker current="sonnet" persona="test" onClose={onClose} />,
    );

    fireEvent.click(screen.getByLabelText(/opus/i));
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    await waitFor(() => {
      expect(onClose).toHaveBeenCalledWith("opus");
    });
  });

  it("calls onClose with no argument when Cancel is clicked", () => {
    const onClose = vi.fn();
    render(
      <ModelPicker current="sonnet" persona="test" onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledWith();
  });

  it("shows an error alert when setPersonaModel fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: false, error: "unknown_model" }), { status: 400 }),
      ),
    );

    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );

    fireEvent.click(screen.getByLabelText(/opus/i));
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
  });
});
