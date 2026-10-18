import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { StepVoiceTemplate } from "./StepVoiceTemplate";

function renderStep(chatModel: string, overrides: Partial<Parameters<typeof StepVoiceTemplate>[0]> = {}) {
  const props = {
    step: 5,
    totalSteps: 7,
    template: "default" as const,
    onTemplateChange: vi.fn(),
    chatModel,
    onChatModelChange: vi.fn(),
    onNext: vi.fn(),
    onBack: vi.fn(),
    avatar: null,
    ...overrides,
  };
  render(<StepVoiceTemplate {...props} />);
  return props;
}

describe("StepVoiceTemplate — optional Claude model (#441)", () => {
  afterEach(cleanup);

  it("tucks the model field under Advanced, collapsed while blank, with the honest hint", () => {
    renderStep("");
    expect(screen.getByText("Advanced").closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText(/Checked when first used/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).not.toBeDisabled();
  });

  it("reports what is typed", () => {
    const p = renderStep("");
    fireEvent.change(screen.getByLabelText("Claude model"), { target: { value: "claude-sonnet-5" } });
    expect(p.onChatModelChange).toHaveBeenCalledWith("claude-sonnet-5");
  });

  it("opens by itself when a model is already set and lets a valid id continue", () => {
    const p = renderStep("claude-sonnet-5");
    expect(screen.getByText("Advanced").closest("details")).toHaveAttribute("open");
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(p.onNext).toHaveBeenCalled();
  });

  it("blocks Continue and shows the format hint for an invalid value", () => {
    const p = renderStep("Claude-Sonnet-5");
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(/start with claude-/i);
    expect(p.onNext).not.toHaveBeenCalled();
  });

  it("makes no network or Tauri call (syntax only)", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    renderStep("claude-sonnet-5");
    fireEvent.change(screen.getByLabelText("Claude model"), { target: { value: "claude-opus-4-8" } });
    expect(fetchSpy).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
