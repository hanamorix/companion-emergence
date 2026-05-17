import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Mock fetchPersonaFeed BEFORE importing the component.
const fetchPersonaFeed = vi.fn();
vi.mock("../../bridge", () => ({
  fetchPersonaFeed: (...args: unknown[]) => fetchPersonaFeed(...args),
}));

import { FeedPanel } from "./FeedPanel";

// Minimal PersonaState shape — only the fields FeedPanel reads.
function fakeState(persona = "test") {
  return { persona } as any;
}

describe("FeedPanel", () => {
  beforeEach(() => {
    fetchPersonaFeed.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows 'No signal yet.' when state is null", () => {
    fetchPersonaFeed.mockResolvedValue([]);
    render(<FeedPanel state={null} />);
    expect(screen.getByText(/No signal yet\./)).toBeInTheDocument();
  });

  it("shows 'Quiet inside.' when feed is empty but state is non-null", async () => {
    fetchPersonaFeed.mockResolvedValue([]);
    render(<FeedPanel state={fakeState()} />);
    await waitFor(() => expect(screen.getByText(/Quiet inside\./)).toBeInTheDocument());
  });

  it("renders a dream entry with opener italicized and type label uppercase", async () => {
    fetchPersonaFeed.mockResolvedValue([
      {
        type: "dream",
        ts: new Date().toISOString(),
        opener: "I dreamed",
        body: "about a lighthouse at the edge.",
        audit_id: null,
      },
    ]);
    render(<FeedPanel state={fakeState()} />);
    await waitFor(() => {
      expect(screen.getByText(/I dreamed/)).toBeInTheDocument();
    });
    expect(screen.getByText(/about a lighthouse at the edge/)).toBeInTheDocument();
    expect(screen.getByText(/Dream/)).toBeInTheDocument(); // type label
  });

  it("renders all 5 type labels when entries of every kind are present", async () => {
    const now = new Date().toISOString();
    fetchPersonaFeed.mockResolvedValue([
      { type: "dream", ts: now, opener: "I dreamed", body: "d", audit_id: null },
      { type: "research", ts: now, opener: "I've been researching", body: "r", audit_id: null },
      { type: "soul", ts: now, opener: "I noticed", body: "s", audit_id: null },
      { type: "outreach", ts: now, opener: "I reached out", body: "o", audit_id: "ia_x" },
      { type: "voice_edit", ts: now, opener: "I wanted to change", body: "v", audit_id: "ia_y" },
    ]);
    render(<FeedPanel state={fakeState()} />);
    for (const label of ["Dream", "Research", "Soul", "Outreach", "Voice edit"]) {
      await waitFor(() => expect(screen.getByText(label)).toBeInTheDocument());
    }
  });

  it("shows the fresh-pulse marker for entries within the last 5 minutes", async () => {
    const recent = new Date().toISOString();
    fetchPersonaFeed.mockResolvedValue([
      { type: "dream", ts: recent, opener: "I dreamed", body: "fresh dream", audit_id: null },
    ]);
    const { container } = render(<FeedPanel state={fakeState()} />);
    await waitFor(() => {
      expect(container.querySelector('[data-fresh="true"]')).toBeInTheDocument();
    });
  });

  it("omits the fresh-pulse marker for older entries", async () => {
    const old = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(); // 24h ago
    fetchPersonaFeed.mockResolvedValue([
      { type: "dream", ts: old, opener: "I dreamed", body: "old dream", audit_id: null },
    ]);
    const { container } = render(<FeedPanel state={fakeState()} />);
    await waitFor(() => {
      expect(container.querySelector('[data-fresh="true"]')).not.toBeInTheDocument();
    });
  });
});
