import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { cleanup, render, screen, waitFor, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Mock fetchAttunement BEFORE importing the component.
const fetchAttunement = vi.fn();
vi.mock("../../bridge", () => ({
  fetchAttunement: (...args: unknown[]) => fetchAttunement(...args),
}));

import { AttunementPanel } from "./AttunementPanel";
import type { AttunementPayload } from "../../bridge";

function emptyPayload(): AttunementPayload {
  return { current_read: null, learned_patterns: [], backfill: null };
}

function makeCurrentRead(overrides: Partial<AttunementPayload["current_read"] & {}> = {}) {
  return {
    ts: new Date().toISOString(),
    source_turn_id: "turn_1",
    tone_label: "warm",
    tone_justification: "uses soft diminutives",
    cadence_label: "unhurried",
    cadence_justification: "long thoughtful sentences",
    mood_valence: 0.7,
    mood_intensity: 0.5,
    predicted_arc_shape: "settling into something",
    schema_version: "1",
    ...overrides,
  };
}

function makePattern(
  overrides: Partial<import("../../bridge").LearnedPattern> = {},
): import("../../bridge").LearnedPattern {
  return {
    id: "pat_1",
    category: "tone",
    canonical_key: "warm_tone",
    description: "Tends to write with warmth",
    evidence_count: 5,
    maturity: "known",
    first_seen_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    last_confirmed_at: new Date().toISOString(),
    last_addressed_at: null,
    crystallised_at: null,
    falsified_at: null,
    examples: [],
    schema_version: "1",
    ...overrides,
  };
}

describe("AttunementPanel", () => {
  beforeEach(() => {
    fetchAttunement.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  // ── Empty state ──────────────────────────────────────────────────────────

  it("shows gentle empty state when payload has no content", async () => {
    fetchAttunement.mockResolvedValue(emptyPayload());
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/Nothing yet/i)).toBeInTheDocument(),
    );
  });

  it("does not show empty state when current_read is present", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead(),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.queryByText(/Nothing yet/i)).not.toBeInTheDocument(),
    );
  });

  // ── Current read section ─────────────────────────────────────────────────

  it("renders 'Right now' section heading when current_read is present", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead(),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/Right now/i)).toBeInTheDocument(),
    );
  });

  it("renders tone_label in the current read paragraph", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead({ tone_label: "pensive" }),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/pensive/i)).toBeInTheDocument(),
    );
  });

  it("renders tone_justification in the current read paragraph", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead({ tone_justification: "uses careful word choices" }),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/uses careful word choices/i)).toBeInTheDocument(),
    );
  });

  it("renders cadence_label in the current read paragraph", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead({ cadence_label: "staccato" }),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/staccato/i)).toBeInTheDocument(),
    );
  });

  it("renders predicted_arc_shape when present", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead({ predicted_arc_shape: "building toward something" }),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/building toward something/i)).toBeInTheDocument(),
    );
  });

  it("omits the arc shape line when predicted_arc_shape is empty", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead({ predicted_arc_shape: "" }),
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.queryByText(/Where this seems/i)).not.toBeInTheDocument(),
    );
  });

  // ── "What she's come to know" section ───────────────────────────────────

  it("renders 'What she's come to know' section heading when surfaceable patterns exist", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ maturity: "known" })],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/What she.s come to know/i)).toBeInTheDocument(),
    );
  });

  it("renders pattern description", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ description: "Prefers short, punchy sentences" })],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/Prefers short, punchy sentences/i)).toBeInTheDocument(),
    );
  });

  it("renders category badge on a pattern", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ category: "cadence", maturity: "forming" })],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/cadence/i)).toBeInTheDocument(),
    );
  });

  it("renders maturity badge on a pattern", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ maturity: "forming" })],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/forming/i)).toBeInTheDocument(),
    );
  });

  it("renders falsified patterns with line-through style", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ maturity: "falsified", description: "Was thought to prefer lists" })],
    });
    const { container } = render(<AttunementPanel persona="test" />);
    await waitFor(() => {
      expect(screen.getByText(/Was thought to prefer lists/i)).toBeInTheDocument();
    });
    const item = container.querySelector("[data-maturity='falsified']");
    expect(item).toBeInTheDocument();
  });

  it("hides immature patterns by default", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ maturity: "immature", description: "Nascent pattern not yet visible" })],
    });
    render(<AttunementPanel persona="test" />);
    // Wait for the fetch to resolve
    await waitFor(() => fetchAttunement.mock.calls.length > 0);
    // Small settle — immature should stay hidden
    expect(screen.queryByText(/Nascent pattern not yet visible/i)).not.toBeInTheDocument();
  });

  it("shows 'known' patterns before 'forming' patterns (sort by maturity desc)", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [
        makePattern({ id: "pat_forming", maturity: "forming", description: "Forming pattern" }),
        makePattern({ id: "pat_known", maturity: "known", description: "Known pattern" }),
      ],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() => {
      expect(screen.getByText(/Known pattern/i)).toBeInTheDocument();
      expect(screen.getByText(/Forming pattern/i)).toBeInTheDocument();
    });
    const items = screen.getAllByRole("listitem");
    const knownIdx = items.findIndex((el) => el.textContent?.includes("Known pattern"));
    const formingIdx = items.findIndex((el) => el.textContent?.includes("Forming pattern"));
    expect(knownIdx).toBeLessThan(formingIdx);
  });

  it("renders examples disclosure button when examples are present", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [
        makePattern({
          maturity: "known",
          examples: ["She said 'darling' three times in a row"],
        }),
      ],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/Examples/i)).toBeInTheDocument(),
    );
  });

  it("renders examples content after expanding disclosure", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [
        makePattern({
          maturity: "known",
          examples: ["She said 'darling' three times in a row"],
        }),
      ],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() => screen.getByText(/Examples/i));
    const summary = screen.getByText(/Examples/i);
    fireEvent.click(summary);
    expect(screen.getByText(/She said 'darling' three times in a row/i)).toBeInTheDocument();
  });

  it("shows 'Named' label when last_addressed_at is not null", async () => {
    const addressedAt = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString();
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ last_addressed_at: addressedAt })],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/Named:/i)).toBeInTheDocument(),
    );
  });

  it("omits 'Named' label when last_addressed_at is null", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern({ last_addressed_at: null })],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() => screen.getByText(/Tends to write with warmth/i));
    expect(screen.queryByText(/Named:/i)).not.toBeInTheDocument();
  });

  it("renders 'First noticed' footer on each pattern", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      learned_patterns: [makePattern()],
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/First noticed:/i)).toBeInTheDocument(),
    );
  });

  // ── Backfill banner ──────────────────────────────────────────────────────

  it("shows backfill banner when backfill.status is 'running'", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead(),
      backfill: {
        started_at: new Date().toISOString(),
        total_windows: 100,
        sampled_windows: 42,
        processed_windows: 42,
        patterns_emitted: 3,
        status: "running",
        last_cursor: "",
        schema_version: "1",
      },
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/is getting to know you/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/42 \/ 100/i)).toBeInTheDocument();
  });

  it("hides backfill banner when backfill.status is 'complete'", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      current_read: makeCurrentRead(),
      backfill: {
        started_at: new Date().toISOString(),
        total_windows: 100,
        sampled_windows: 100,
        processed_windows: 100,
        patterns_emitted: 8,
        status: "complete",
        last_cursor: "",
        schema_version: "1",
      },
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() => screen.getByText(/Right now/i));
    expect(screen.queryByText(/is getting to know you/i)).not.toBeInTheDocument();
  });

  it("shows backfill banner when backfill.status is 'interrupted'", async () => {
    fetchAttunement.mockResolvedValue({
      ...emptyPayload(),
      backfill: {
        started_at: new Date().toISOString(),
        total_windows: 50,
        sampled_windows: 10,
        processed_windows: 10,
        patterns_emitted: 1,
        status: "interrupted",
        last_cursor: "",
        schema_version: "1",
      },
    });
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/is getting to know you/i)).toBeInTheDocument(),
    );
  });

  // ── Error state ──────────────────────────────────────────────────────────

  it("shows error message when fetch fails", async () => {
    fetchAttunement.mockRejectedValue(new Error("bridge unavailable"));
    render(<AttunementPanel persona="test" />);
    await waitFor(() =>
      expect(screen.getByText(/bridge unavailable/i)).toBeInTheDocument(),
    );
  });

  // ── persona-name interpolation (SITE 2 & 3) ─────────────────────────────

  it("shows companion name (not 'Nell') in empty-state copy when persona is 'mira'", async () => {
    fetchAttunement.mockResolvedValue(emptyPayload());
    render(<AttunementPanel persona="mira" />);
    await waitFor(() =>
      expect(screen.getByText(/Mira/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText(/\bNell\b/)).not.toBeInTheDocument();
  });
});
