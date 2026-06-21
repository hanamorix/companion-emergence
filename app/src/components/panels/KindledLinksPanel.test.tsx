import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { afterEach, describe, expect, it, vi } from "vitest";
import { KindledLinksPanel } from "./KindledLinksPanel";

function makePeer(consent_state: string) {
  return {
    peer_id: "kid_a",
    fingerprint: "kid_a",
    relay_url: "https://r",
    consent_state,
    stage: "familiar",
    affinity_tags: ["dreams"],
    has_active_session: false,
  };
}

vi.mock("../../bridge", () => ({
  fetchKindledPeers: vi.fn(async () => [makePeer("paired")]),
  fetchKindledHolds: vi.fn(async () => ({ held_count: 0, items: [] })),
  fetchKindledTranscript: vi.fn(async () => []),
  createKindledInvite: vi.fn(), acceptKindledInvite: vi.fn(), setKindledConsent: vi.fn(),
}));

afterEach(cleanup);

describe("KindledLinksPanel", () => {
  it("renders a peer with stage + consent and no compose box", async () => {
    render(<KindledLinksPanel persona="nell" />);
    expect(await screen.findByText(/familiar/)).toBeInTheDocument();
    // no-typing guarantee: no message-compose textarea/input for the peer convo
    expect(screen.queryByRole("textbox", { name: /message|reply|compose/i })).toBeNull();
  });

  it("paired peer shows Pause + Revoke + Block, no Resume", async () => {
    const { fetchKindledPeers } = await import("../../bridge");
    vi.mocked(fetchKindledPeers).mockResolvedValue([makePeer("paired")]);
    render(<KindledLinksPanel persona="nell" />);
    expect(await screen.findByRole("button", { name: /pause kid_a/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /revoke kid_a/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /block kid_a/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /resume kid_a/i })).toBeNull();
  });

  it("blocked peer renders no consent action buttons", async () => {
    const { fetchKindledPeers } = await import("../../bridge");
    vi.mocked(fetchKindledPeers).mockResolvedValue([makePeer("blocked")]);
    render(<KindledLinksPanel persona="nell" />);
    // wait for the peer row to appear (consent_state badge)
    expect(await screen.findByText(/blocked/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /pause|resume|revoke|block/i })).toBeNull();
  });
});
