import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi } from "vitest";
import { KindledLinksPanel } from "./KindledLinksPanel";
vi.mock("../../bridge", () => ({
  fetchKindledPeers: vi.fn(async () => [{ peer_id: "kid_a", fingerprint: "kid_a",
    relay_url: "https://r", consent_state: "paired", stage: "familiar",
    affinity_tags: ["dreams"], has_active_session: false }]),
  fetchKindledHolds: vi.fn(async () => ({ held_count: 0, items: [] })),
  fetchKindledTranscript: vi.fn(async () => []),
  createKindledInvite: vi.fn(), acceptKindledInvite: vi.fn(), setKindledConsent: vi.fn(),
}));

describe("KindledLinksPanel", () => {
  it("renders a peer with stage + consent and no compose box", async () => {
    render(<KindledLinksPanel persona="nell" />);
    expect(await screen.findByText(/familiar/)).toBeInTheDocument();
    // no-typing guarantee: no message-compose textarea/input for the peer convo
    expect(screen.queryByRole("textbox", { name: /message|reply|compose/i })).toBeNull();
  });
});
