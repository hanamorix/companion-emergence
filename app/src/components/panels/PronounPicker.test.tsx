// PronounPicker — standalone component tests.
// Mirrors ModelPicker.test.tsx pattern.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, _args: { persona: string }) => {
    if (cmd === "get_bridge_credentials") {
      return { port: 50000, auth_token: null };
    }
    throw new Error(`unexpected cmd ${cmd}`);
  }),
}));

import { PronounPicker } from "./PronounPicker";

describe("PronounPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders all three pronoun options", () => {
    render(
      <PronounPicker current="she/her" persona="test" onClose={() => {}} />,
    );

    expect(screen.getByLabelText(/she\/her/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/he\/him/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/they\/them/i)).toBeInTheDocument();
  });

  it("checks the current prop by default", () => {
    render(
      <PronounPicker current="he/him" persona="test" onClose={() => {}} />,
    );
    const radio = screen.getByLabelText(/he\/him/i) as HTMLInputElement;
    expect(radio.checked).toBe(true);
  });

  it("Apply button is disabled when the selection matches current", () => {
    render(
      <PronounPicker current="she/her" persona="test" onClose={() => {}} />,
    );
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).toBeDisabled();
  });

  it("Apply button becomes enabled after selecting a different option", () => {
    render(
      <PronounPicker current="she/her" persona="test" onClose={() => {}} />,
    );
    fireEvent.click(screen.getByLabelText(/he\/him/i));
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).not.toBeDisabled();
  });

  it("calls setPersonaPronouns and onClose with the new preset after apply", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: true }), { status: 200 }),
      ),
    );

    const onClose = vi.fn();
    render(
      <PronounPicker current="she/her" persona="test" onClose={onClose} />,
    );

    fireEvent.click(screen.getByLabelText(/he\/him/i));
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    await waitFor(() => {
      expect(onClose).toHaveBeenCalledWith("he/him");
    });
  });

  it("calls onClose with no argument when Cancel is clicked", () => {
    const onClose = vi.fn();
    render(
      <PronounPicker current="she/her" persona="test" onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledWith();
  });

  it("shows an error alert when setPersonaPronouns fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: false, error: "unknown_preset" }), { status: 400 }),
      ),
    );

    render(
      <PronounPicker current="she/her" persona="test" onClose={() => {}} />,
    );

    fireEvent.click(screen.getByLabelText(/he\/him/i));
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
  });
});
