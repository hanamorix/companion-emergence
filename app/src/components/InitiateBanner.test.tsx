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
    render(<InitiateBanner message={baseMessage} onReply={vi.fn()} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    expect(screen.getByText(/landed somewhere/)).toBeInTheDocument();
  });

  it("calls onMounted exactly once after a brief on-screen delay", async () => {
    vi.useFakeTimers();
    const onMounted = vi.fn();
    render(<InitiateBanner message={baseMessage} onReply={vi.fn()} onDismiss={vi.fn()} onMounted={onMounted} />);
    expect(onMounted).not.toHaveBeenCalled();
    vi.advanceTimersByTime(2100);
    expect(onMounted).toHaveBeenCalledTimes(1);
    expect(onMounted).toHaveBeenCalledWith("ia_001");
    vi.useRealTimers();
  });

  it("emits onReply when the ↩ button is clicked", () => {
    const onReply = vi.fn();
    render(<InitiateBanner message={baseMessage} onReply={onReply} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /reply|↩/i }));
    expect(onReply).toHaveBeenCalledWith("ia_001");
  });

  it("emits onDismiss when the close button is clicked", () => {
    const onDismiss = vi.fn();
    render(<InitiateBanner message={baseMessage} onReply={vi.fn()} onDismiss={onDismiss} onMounted={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss|close/i }));
    expect(onDismiss).toHaveBeenCalledWith("ia_001");
  });

  it("shows state badge reflecting the current state", () => {
    render(<InitiateBanner message={{ ...baseMessage, state: "acknowledged_unclear" }} onReply={vi.fn()} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    expect(screen.getByText(/unclear/i)).toBeInTheDocument();
  });
});
