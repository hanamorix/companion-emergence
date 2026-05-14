import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { VoiceEditPanel } from "./VoiceEditPanel";

afterEach(() => {
  cleanup();
});

describe("VoiceEditPanel", () => {
  const proposal = {
    auditId: "ia_ve_001",
    oldText: "I'm fine when tired",
    newText: "I get quieter when tired",
    rationale: "the old wording felt too clipped",
    evidence: ["dream_a", "cryst_b", "tone_c"],
    voiceTemplate: "line A\nI'm fine when tired\nline C\n",
  };

  it("renders the proposed change in context with surrounding lines", () => {
    render(<VoiceEditPanel proposal={proposal} onAccept={vi.fn()} onReject={vi.fn()} />);
    expect(screen.getByText(/line A/)).toBeInTheDocument();
    expect(screen.getByText(/I'm fine when tired/)).toBeInTheDocument();
    expect(screen.getByText(/I get quieter when tired/)).toBeInTheDocument();
    expect(screen.getByText(/line C/)).toBeInTheDocument();
  });

  it("calls onAccept with null when Accept clicked", () => {
    const onAccept = vi.fn();
    render(<VoiceEditPanel proposal={proposal} onAccept={onAccept} onReject={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /^accept$/i }));
    expect(onAccept).toHaveBeenCalledWith("ia_ve_001", null);
  });

  it("calls onAccept with edited text when Accept with edits is used", () => {
    const onAccept = vi.fn();
    render(<VoiceEditPanel proposal={proposal} onAccept={onAccept} onReject={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /accept with edits/i }));
    const ta = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "tweaked line" } });
    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
    expect(onAccept).toHaveBeenCalledWith("ia_ve_001", "tweaked line");
  });

  it("calls onReject when Reject clicked", () => {
    const onReject = vi.fn();
    render(<VoiceEditPanel proposal={proposal} onAccept={vi.fn()} onReject={onReject} />);
    fireEvent.click(screen.getByRole("button", { name: /reject/i }));
    expect(onReject).toHaveBeenCalledWith("ia_ve_001");
  });
});
