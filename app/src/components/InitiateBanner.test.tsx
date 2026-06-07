import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { InitiateBanner } from "./InitiateBanner";

afterEach(() => {
  cleanup();
});

describe("InitiateBanner", () => {
  const baseMessage = {
    auditId: "ia_001",
    body: "the dream from this morning landed somewhere",
    urgency: "quiet" as const,
    state: "delivered" as const,
    timestamp: "2026-05-11T14:32:00+00:00",
  };

  it("renders the message body", () => {
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    expect(screen.getByText(/landed somewhere/)).toBeInTheDocument();
  });

  it("calls onMounted exactly once after a brief on-screen delay", async () => {
    vi.useFakeTimers();
    const onMounted = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={vi.fn()} onMounted={onMounted} />);
    expect(onMounted).not.toHaveBeenCalled();
    vi.advanceTimersByTime(2100);
    expect(onMounted).toHaveBeenCalledTimes(1);
    expect(onMounted).toHaveBeenCalledWith("ia_001");
    vi.useRealTimers();
  });

  it("renders a reply textarea and sends its text via onSendReply", () => {
    const onSendReply = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={onSendReply} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    const box = screen.getByPlaceholderText(/reply/i);
    fireEvent.change(box, { target: { value: "I hear you, love" } });
    fireEvent.click(screen.getByRole("button", { name: /send reply/i }));
    expect(onSendReply).toHaveBeenCalledWith("ia_001", "I hear you, love");
  });

  it("sends on Enter, newlines on Shift+Enter", () => {
    const onSendReply = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={onSendReply} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    const box = screen.getByPlaceholderText(/reply/i);
    fireEvent.change(box, { target: { value: "hi" } });
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
    expect(onSendReply).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onSendReply).toHaveBeenCalledWith("ia_001", "hi");
  });

  it("does not send empty/whitespace replies", () => {
    const onSendReply = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={onSendReply} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    fireEvent.keyDown(screen.getByPlaceholderText(/reply/i), { key: "Enter" });
    expect(onSendReply).not.toHaveBeenCalled();
  });

  it("emits onDismiss when the close button is clicked", () => {
    const onDismiss = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={onDismiss} onMounted={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalledWith("ia_001");
  });

  it("shows the 'reached out' header with the companion name", () => {
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    expect(screen.getByText(/Nell reached out/i)).toBeInTheDocument();
  });

  it("does not call onMounted when document is hidden at mount time", () => {
    vi.useFakeTimers();
    const hiddenSpy = vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    const onMounted = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={vi.fn()} onMounted={onMounted} />);
    vi.advanceTimersByTime(5000);
    expect(onMounted).not.toHaveBeenCalled();
    hiddenSpy.mockRestore();
    vi.useRealTimers();
  });

  it("pauses timer when document becomes hidden mid-countdown and resumes when visible again", () => {
    vi.useFakeTimers();
    const hiddenSpy = vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    const onMounted = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={vi.fn()} onMounted={onMounted} />);
    // 1 second elapses (not enough to fire).
    vi.advanceTimersByTime(1000);
    expect(onMounted).not.toHaveBeenCalled();
    // Document becomes hidden — timer should be cleared.
    hiddenSpy.mockReturnValue(true);
    document.dispatchEvent(new Event("visibilitychange"));
    // Plenty of time passes while hidden; should not fire.
    vi.advanceTimersByTime(10_000);
    expect(onMounted).not.toHaveBeenCalled();
    // Document becomes visible — fresh 2-second timer schedules.
    hiddenSpy.mockReturnValue(false);
    document.dispatchEvent(new Event("visibilitychange"));
    vi.advanceTimersByTime(2100);
    expect(onMounted).toHaveBeenCalledTimes(1);
    expect(onMounted).toHaveBeenCalledWith("ia_001");
    hiddenSpy.mockRestore();
    vi.useRealTimers();
  });

  it("aria-label uses companionName prop, not hardcoded 'Nell'", () => {
    render(
      <InitiateBanner
        message={baseMessage}
        companionName="Mira"
        onSendReply={vi.fn()}
        onDismiss={vi.fn()}
        onMounted={vi.fn()}
      />,
    );
    const region = screen.getByRole("region");
    expect(region).toHaveAttribute("aria-label", "Mira reached out");
    expect(region.getAttribute("aria-label")).not.toContain("Nell");
  });

  it("disables textarea and send button when isStreaming=true, but dismiss stays enabled", () => {
    const onSendReply = vi.fn();
    const onDismiss = vi.fn();
    render(
      <InitiateBanner
        message={baseMessage}
        companionName="Nell"
        onSendReply={onSendReply}
        onDismiss={onDismiss}
        onMounted={vi.fn()}
        isStreaming={true}
      />,
    );
    const textarea = screen.getByPlaceholderText(/reply/i);
    const sendBtn = screen.getByRole("button", { name: /send reply/i });
    const dismissBtn = screen.getByRole("button", { name: /dismiss/i });

    expect(textarea).toBeDisabled();
    expect(sendBtn).toBeDisabled();
    expect(dismissBtn).not.toBeDisabled();
  });

  it("does not call onSendReply via Enter or click while isStreaming=true", () => {
    const onSendReply = vi.fn();
    render(
      <InitiateBanner
        message={baseMessage}
        companionName="Nell"
        onSendReply={onSendReply}
        onDismiss={vi.fn()}
        onMounted={vi.fn()}
        isStreaming={true}
      />,
    );
    const textarea = screen.getByPlaceholderText(/reply/i);
    const sendBtn = screen.getByRole("button", { name: /send reply/i });
    fireEvent.click(sendBtn);
    expect(onSendReply).not.toHaveBeenCalled();
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter" });
    expect(onSendReply).not.toHaveBeenCalled();
  });

  it("calls onDismiss when × is clicked even while isStreaming=true", () => {
    const onDismiss = vi.fn();
    render(
      <InitiateBanner
        message={baseMessage}
        companionName="Nell"
        onSendReply={vi.fn()}
        onDismiss={onDismiss}
        onMounted={vi.fn()}
        isStreaming={true}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalledWith("ia_001");
  });

  it("does not fire onMounted twice across multiple visibility cycles", () => {
    vi.useFakeTimers();
    const hiddenSpy = vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    const onMounted = vi.fn();
    render(<InitiateBanner message={baseMessage} companionName="Nell" onSendReply={vi.fn()} onDismiss={vi.fn()} onMounted={onMounted} />);
    vi.advanceTimersByTime(2100);
    expect(onMounted).toHaveBeenCalledTimes(1);
    // Hide + show again — should not re-fire.
    hiddenSpy.mockReturnValue(true);
    document.dispatchEvent(new Event("visibilitychange"));
    hiddenSpy.mockReturnValue(false);
    document.dispatchEvent(new Event("visibilitychange"));
    vi.advanceTimersByTime(5000);
    expect(onMounted).toHaveBeenCalledTimes(1);
    hiddenSpy.mockRestore();
    vi.useRealTimers();
  });
});
