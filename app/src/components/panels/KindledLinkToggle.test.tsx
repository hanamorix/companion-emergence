// KindledLinkToggle — component tests (mirrors NotesToggle.test.tsx pattern).

import { describe, test, expect, vi, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { KindledLinkToggle } from "./KindledLinkToggle";
import * as bridge from "../../bridge";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("KindledLinkToggle", () => {
  test("renders off by default with descriptive copy", () => {
    vi.spyOn(bridge, "setKindledLinkEnabled").mockResolvedValue({
      kindled_link_enabled: false,
      kindled_relay_url: null,
    });
    render(
      <KindledLinkToggle
        persona="nell"
        enabled={false}
        relayUrl={null}
      />,
    );
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).not.toBeChecked();
    // copy must mention correspondence (autonomous network)
    expect(screen.getByText(/correspond/i)).toBeInTheDocument();
  });

  test("toggling on calls setKindledLinkEnabled with enabled=true and null relay", async () => {
    vi.spyOn(bridge, "setKindledLinkEnabled").mockResolvedValue({
      kindled_link_enabled: true,
      kindled_relay_url: null,
    });
    render(
      <KindledLinkToggle
        persona="nell"
        enabled={false}
        relayUrl={null}
      />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    await waitFor(() =>
      expect(bridge.setKindledLinkEnabled).toHaveBeenCalledWith("nell", true, null),
    );
  });

  test("relay URL input is visible when enabled=true and persists on blur", async () => {
    vi.spyOn(bridge, "setKindledLinkEnabled").mockResolvedValue({
      kindled_link_enabled: true,
      kindled_relay_url: "http://127.0.0.1:9000",
    });
    render(
      <KindledLinkToggle
        persona="nell"
        enabled={true}
        relayUrl={null}
      />,
    );
    const input = screen.getByRole("textbox");
    expect(input).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "http://127.0.0.1:9000" } });
    fireEvent.blur(input);
    await waitFor(() =>
      expect(bridge.setKindledLinkEnabled).toHaveBeenCalledWith(
        "nell",
        true,
        "http://127.0.0.1:9000",
      ),
    );
  });
});
