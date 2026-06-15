import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { PendingWriteCard } from "./components/PendingWriteCard";

afterEach(() => {
  cleanup();
});

const baseWrite = {
  id: "x",
  op: "create" as const,
  path: "/Users/h/n.md",
  preview: "hello",
  truncated: false,
  proposed_at: "2026-06-14T12:00:00+00:00",
};

describe("PendingWriteCard", () => {
  it("renders op/path/preview and calls approve", () => {
    const approve = vi.fn();
    render(
      <PendingWriteCard write={baseWrite} onApprove={approve} onDecline={vi.fn()} />,
    );
    expect(screen.getByText(/n\.md/)).toBeInTheDocument();
    expect(screen.getByText(/hello/)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/approve/i));
    expect(approve).toHaveBeenCalledWith("x");
  });

  it("calls decline with the id", () => {
    const decline = vi.fn();
    render(
      <PendingWriteCard write={baseWrite} onApprove={vi.fn()} onDecline={decline} />,
    );
    fireEvent.click(screen.getByText(/decline/i));
    expect(decline).toHaveBeenCalledWith("x");
  });

  it("shows the op badge", () => {
    render(
      <PendingWriteCard write={baseWrite} onApprove={vi.fn()} onDecline={vi.fn()} />,
    );
    expect(screen.getByText(/create/i)).toBeInTheDocument();
  });

  it("shows a (truncated) marker when truncated", () => {
    render(
      <PendingWriteCard
        write={{ ...baseWrite, truncated: true }}
        onApprove={vi.fn()}
        onDecline={vi.fn()}
      />,
    );
    expect(screen.getByText(/truncated/i)).toBeInTheDocument();
  });

  it("does NOT show the truncated marker when not truncated", () => {
    render(
      <PendingWriteCard write={baseWrite} onApprove={vi.fn()} onDecline={vi.fn()} />,
    );
    expect(screen.queryByText(/truncated/i)).not.toBeInTheDocument();
  });

  it("disables both buttons while busy", () => {
    render(
      <PendingWriteCard
        write={baseWrite}
        onApprove={vi.fn()}
        onDecline={vi.fn()}
        busy={true}
      />,
    );
    expect(screen.getByText(/approve/i)).toBeDisabled();
    expect(screen.getByText(/decline/i)).toBeDisabled();
  });
});
