import { describe, it, expect, vi, afterEach, beforeAll } from "vitest";
import { render, screen, waitFor, fireEvent, cleanup } from "@testing-library/react";
import { GalleryPanel } from "./GalleryPanel";

// ── jsdom polyfills ────────────────────────────────────────────────

beforeAll(() => {
  (globalThis as any).IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

// Clean up after each test — prevents DOM accumulation across tests.
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// ── module mock ────────────────────────────────────────────────────

const { getBridgeCredentials } = vi.hoisted(() => ({
  getBridgeCredentials: vi.fn().mockResolvedValue({
    url: "http://127.0.0.1:50420",
    port: 50420,
    authToken: null,
  }),
}));

vi.mock("../../bridge", () => ({
  getBridgeCredentials,
}));

// ── helpers ────────────────────────────────────────────────────────

type ImageEntry = {
  sha: string;
  ext: string;
  first_seen_ts: string;
  first_8_chars: string;
};

function mk(sha: string): ImageEntry {
  return {
    sha: sha.repeat(64).slice(0, 64),
    ext: "png",
    first_seen_ts: "2026-01-01T00:00:00Z",
    first_8_chars: sha.slice(0, 8),
  };
}

/** Stub fetch globally.  Every call after the first hangs forever so
 *  React StrictMode double-mounts don't trigger unhandled rejections. */
function stubFetch(body: unknown, status = 200) {
  let calls = 0;
  vi.spyOn(globalThis, "fetch").mockImplementation(() => {
    calls++;
    if (calls === 1) {
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }
    return new Promise<Response>(() => {});
  });
}

// ── tests ──────────────────────────────────────────────────────────

describe("GalleryPanel", () => {
  it("shows loading state initially", () => {
    // fetch never resolves → stays in loading forever
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise<Response>(() => {}),
    );
    render(<GalleryPanel persona="test" />);
    expect(screen.getByText(/loading gallery/i)).toBeTruthy();
  });

  it("shows empty state when no images", async () => {
    stubFetch([]);
    render(<GalleryPanel persona="test" />);
    await waitFor(() => {
      expect(screen.getByText(/no images shared yet/i)).toBeTruthy();
    });
  });

  it("renders thumbnail grid when images returned", async () => {
    stubFetch([mk("a"), mk("b")]);
    render(<GalleryPanel persona="test" />);

    await waitFor(() => {
      const buttons = screen.getAllByRole("button", { name: /view image/i });
      expect(buttons).toHaveLength(2);
    });
  });

  it("opens lightbox on thumbnail click", async () => {
    const img = mk("a");
    stubFetch([img]);
    render(<GalleryPanel persona="test" />);

    const btn = await screen.findByLabelText(/view image/i);
    fireEvent.click(btn);
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("closes lightbox on Escape", async () => {
    const img = mk("a");
    stubFetch([img]);
    render(<GalleryPanel persona="test" />);

    const btn = await screen.findByLabelText(/view image/i);
    fireEvent.click(btn);
    expect(screen.getByRole("dialog")).toBeTruthy();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  it("shows error state on fetch failure", async () => {
    stubFetch("nope", 500);
    render(<GalleryPanel persona="test" />);
    await waitFor(() => {
      expect(screen.getByText(/could not load images/i)).toBeTruthy();
    });
  });
});
