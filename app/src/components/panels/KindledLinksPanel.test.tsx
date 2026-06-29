import { cleanup, render, screen, fireEvent, act } from "@testing-library/react";
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
  fetchKindledLinkStatus: vi.fn(async () => ({
    relay_ok: null,
    last_poll_ts: null,
    last_push_ts: null,
    degraded_peers: [],
    recovered: false,
  })),
  createKindledInvite: vi.fn(), acceptKindledInvite: vi.fn(), setKindledConsent: vi.fn(),
  fetchKindledMyCode: vi.fn(async () => ({ code: "test-code", fingerprint_phrase: "a b c" })),
  connectKindled: vi.fn(async () => ({})),
  rotateKindledIdentity: vi.fn(async () => ({ new_key_id: "k1", fingerprint_phrase: "x y z" })),
  runKindledSelfTest: vi.fn(async () => ({
    ok: true,
    relay_url: "https://relay.test",
    stages: [
      { name: "identity", ok: true, detail: "" },
      { name: "relay_connect", ok: true, detail: "" },
    ],
  })),
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

  it("shows recovery banner when status reports recovered=true", async () => {
    const { fetchKindledLinkStatus } = await import("../../bridge");
    vi.mocked(fetchKindledLinkStatus).mockResolvedValue({
      relay_ok: true,
      last_poll_ts: "2026-06-22T10:00:00Z",
      last_push_ts: null,
      degraded_peers: [],
      recovered: true,
    });
    render(<KindledLinksPanel persona="nell" />);
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/recover/i);
  });

  it("shows relay-health line with relay status", async () => {
    const { fetchKindledLinkStatus } = await import("../../bridge");
    vi.mocked(fetchKindledLinkStatus).mockResolvedValue({
      relay_ok: true,
      last_poll_ts: "2026-06-22T10:00:00Z",
      last_push_ts: null,
      degraded_peers: [],
      recovered: false,
    });
    render(<KindledLinksPanel persona="nell" />);
    // relay-health line must be visible somewhere
    expect(await screen.findByText(/relay/i)).toBeInTheDocument();
  });
});

describe("KindledLinksPanel — self-test section", () => {
  it("renders a Test my setup button", async () => {
    render(<KindledLinksPanel persona="nell" />);
    expect(await screen.findByRole("button", { name: /test my setup/i })).toBeInTheDocument();
  });

  it("shows PASS verdict and per-stage rows after a successful self-test", async () => {
    const { runKindledSelfTest } = await import("../../bridge");
    vi.mocked(runKindledSelfTest).mockResolvedValueOnce({
      ok: true,
      relay_url: "https://relay.test",
      stages: [
        { name: "identity", ok: true, detail: "" },
        { name: "relay_connect", ok: true, detail: "" },
      ],
    });
    render(<KindledLinksPanel persona="nell" />);
    const btn = await screen.findByRole("button", { name: /test my setup/i });
    await act(async () => { fireEvent.click(btn); });
    expect(await screen.findByText(/pass/i)).toBeInTheDocument();
    expect(screen.getByText("identity")).toBeInTheDocument();
    expect(screen.getByText("relay_connect")).toBeInTheDocument();
  });

  it("shows FAIL verdict and failing stage detail on a failed self-test", async () => {
    const { runKindledSelfTest } = await import("../../bridge");
    vi.mocked(runKindledSelfTest).mockResolvedValueOnce({
      ok: false,
      relay_url: "https://relay.test",
      stages: [
        { name: "identity", ok: true, detail: "" },
        { name: "relay_connect", ok: false, detail: "connection refused" },
      ],
    });
    render(<KindledLinksPanel persona="nell" />);
    const btn = await screen.findByRole("button", { name: /test my setup/i });
    await act(async () => { fireEvent.click(btn); });
    expect(await screen.findByText(/fail/i)).toBeInTheDocument();
    expect(screen.getByText(/connection refused/i)).toBeInTheDocument();
  });

  it("shows inline error when runKindledSelfTest throws", async () => {
    const { runKindledSelfTest } = await import("../../bridge");
    vi.mocked(runKindledSelfTest).mockRejectedValueOnce(new Error("network error"));
    render(<KindledLinksPanel persona="nell" />);
    const btn = await screen.findByRole("button", { name: /test my setup/i });
    await act(async () => { fireEvent.click(btn); });
    expect(await screen.findByText(/network error/i)).toBeInTheDocument();
  });
});
