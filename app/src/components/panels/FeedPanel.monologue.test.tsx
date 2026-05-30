import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Mock fetchPersonaFeed BEFORE importing the component.
const fetchPersonaFeed = vi.fn();
vi.mock("../../bridge", () => ({
  fetchPersonaFeed: (...args: unknown[]) => fetchPersonaFeed(...args),
}));

import { FeedPanel } from "./FeedPanel";

function fakeState(persona = "test") {
  return { persona } as any;
}

describe("FeedPanel — monologue entries", () => {
  beforeEach(() => {
    fetchPersonaFeed.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("renders the opener", async () => {
    fetchPersonaFeed.mockResolvedValue([
      {
        type: "monologue",
        ts: new Date().toISOString(),
        opener: "what was running underneath",
        body: "she searched for Loopy and felt fond when nothing surfaced",
        audit_id: null,
      },
    ]);
    render(<FeedPanel state={fakeState()} />);
    await waitFor(() =>
      expect(screen.getByText(/what was running underneath/i)).toBeInTheDocument(),
    );
  });

  it("renders the type label for monologue", async () => {
    fetchPersonaFeed.mockResolvedValue([
      {
        type: "monologue",
        ts: new Date().toISOString(),
        opener: "quiet thought",
        body: "something passing through",
        audit_id: null,
      },
    ]);
    render(<FeedPanel state={fakeState()} />);
    await waitFor(() =>
      expect(screen.getByText(/Monologue/i)).toBeInTheDocument(),
    );
  });

  it("renders the body", async () => {
    fetchPersonaFeed.mockResolvedValue([
      {
        type: "monologue",
        ts: new Date().toISOString(),
        opener: "what was running underneath",
        body: "she searched for Loopy and felt fond when nothing surfaced",
        audit_id: null,
      },
    ]);
    render(<FeedPanel state={fakeState()} />);
    await waitFor(() =>
      expect(
        screen.getByText(/she searched for Loopy and felt fond/),
      ).toBeInTheDocument(),
    );
  });

  it("does not regress other entry types when mixed with monologue", async () => {
    const now = new Date().toISOString();
    fetchPersonaFeed.mockResolvedValue([
      {
        type: "dream",
        ts: now,
        opener: "I dreamed",
        body: "about the sea",
        audit_id: null,
      },
      {
        type: "monologue",
        ts: now,
        opener: "what was running underneath",
        body: "she searched for Loopy and felt fond when nothing surfaced",
        audit_id: null,
      },
    ]);
    render(<FeedPanel state={fakeState()} />);
    await waitFor(() => {
      expect(screen.getByText(/I dreamed/)).toBeInTheDocument();
      expect(screen.getByText(/about the sea/)).toBeInTheDocument();
      expect(screen.getByText(/what was running underneath/i)).toBeInTheDocument();
      expect(screen.getByText(/Dream/)).toBeInTheDocument();
      expect(screen.getByText(/Monologue/i)).toBeInTheDocument();
    });
  });
});
