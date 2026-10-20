// ModelPicker — standalone component tests.
// Verifies render, apply action, and cost hint copy.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, _args: { persona: string }) => {
    if (cmd === "get_bridge_credentials") {
      return { port: 50000, auth_token: null };
    }
    throw new Error(`unexpected cmd ${cmd}`);
  }),
}));

import { ModelPicker } from "./ModelPicker";

describe("ModelPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders all three model options with cost hints", () => {
    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );

    expect(screen.getByLabelText(/sonnet/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/opus/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/haiku/i)).toBeInTheDocument();

    // Cost hints are visible
    expect(screen.getByText(/recommended/i)).toBeInTheDocument();
    expect(screen.getByText(/smartest/i)).toBeInTheDocument();
    expect(screen.getByText(/fastest.*cheapest/i)).toBeInTheDocument();
  });

  it("Apply button is disabled when the selection matches current", () => {
    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).toBeDisabled();
  });

  it("Apply button becomes enabled after selecting a different model", () => {
    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );
    fireEvent.click(screen.getByLabelText(/opus/i));
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).not.toBeDisabled();
  });

  it("calls onClose with the new model after a successful apply", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }),
      ),
    );

    const onClose = vi.fn();
    render(
      <ModelPicker current="sonnet" persona="test" onClose={onClose} />,
    );

    fireEvent.click(screen.getByLabelText(/opus/i));
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    await waitFor(() => {
      expect(onClose).toHaveBeenCalledWith("opus");
    });
  });

  it("calls onClose with no argument when Cancel is clicked", () => {
    const onClose = vi.fn();
    render(
      <ModelPicker current="sonnet" persona="test" onClose={onClose} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledWith();
  });

  it("shows an error alert when setPersonaModel fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ok: false, error: "unknown_model" }), { status: 400 }),
      ),
    );

    render(
      <ModelPicker current="sonnet" persona="test" onClose={() => {}} />,
    );

    fireEvent.click(screen.getByLabelText(/opus/i));
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
  });

  // ---- #440: custom model ID -------------------------------------------------------------

  const customRadio = () => screen.getByRole("radio", { name: /custom model id/i });
  const idInput = () => screen.getByRole("textbox", { name: "Model ID" });
  const applyBtn = () => screen.getByRole("button", { name: /apply|checking/i });

  function stubFetch(response: Response) {
    const fetchMock = vi.fn().mockResolvedValue(response);
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("offers a custom model ID option and input beside the three aliases", () => {
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    expect(customRadio()).toBeInTheDocument();
    expect(customRadio()).not.toBeChecked();
    expect(idInput()).toBeInTheDocument();
    expect(idInput()).toHaveAttribute("maxlength", "64");
    expect(screen.getAllByRole("radio")).toHaveLength(4);
  });

  it("sends a valid custom id to the bridge as written and reports it", async () => {
    const fetchMock = stubFetch(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true }), { status: 200 }),
    );
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);

    fireEvent.change(idInput(), { target: { value: "claude-sonnet-5" } });
    expect(customRadio()).toBeChecked();
    fireEvent.click(applyBtn());

    await waitFor(() => expect(onClose).toHaveBeenCalledWith("claude-sonnet-5"));
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toContain("/persona/config/model");
    expect(JSON.parse(init!.body as string)).toEqual({ model: "claude-sonnet-5" });
  });

  it.each(["Claude-Sonnet-5", "sonnet-5", "claude-", "claude-sonnet 5", "claude_sonnet_5"])(
    "disables Apply and shows the format hint for %j",
    (bad) => {
      render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
      fireEvent.change(idInput(), { target: { value: bad } });
      expect(applyBtn()).toBeDisabled();
      expect(screen.getByText(/start with claude-/i)).toBeInTheDocument();
    },
  );

  it("shows no hint before anything is typed and keeps Apply disabled", () => {
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    fireEvent.click(customRadio());
    expect(applyBtn()).toBeDisabled();
    expect(screen.queryByText(/start with claude-/i)).not.toBeInTheDocument();
  });

  it("trims surrounding whitespace before validating and sending", async () => {
    const fetchMock = stubFetch(
      new Response(JSON.stringify({ ok: true, model: "claude-opus-4-8", checked: true }), { status: 200 }),
    );
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);

    fireEvent.change(idInput(), { target: { value: "  claude-opus-4-8  " } });
    expect(applyBtn()).not.toBeDisabled();
    fireEvent.click(applyBtn());

    await waitFor(() => expect(onClose).toHaveBeenCalledWith("claude-opus-4-8"));
    expect(JSON.parse(fetchMock.mock.calls[0]![1]!.body as string)).toEqual({ model: "claude-opus-4-8" });
  });

  it("opens with Custom selected and the id filled in when the current model is a full id", () => {
    render(<ModelPicker current="claude-sonnet-5" persona="test" onClose={() => {}} />);
    expect(customRadio()).toBeChecked();
    expect(idInput()).toHaveValue("claude-sonnet-5");
    expect(applyBtn()).toBeDisabled(); // unchanged
    fireEvent.change(idInput(), { target: { value: "claude-sonnet-5-b" } });
    expect(applyBtn()).not.toBeDisabled();
  });

  it("selecting Custom by focusing the input unchecks the aliases", () => {
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    fireEvent.focus(idInput());
    expect(customRadio()).toBeChecked();
    expect(screen.getByLabelText(/sonnet/i)).not.toBeChecked();
  });

  it("shows the bridge's specific refusal, stays open, keeps the typed id and re-enables Apply", async () => {
    stubFetch(
      new Response(
        JSON.stringify({ ok: false, error: "model_unavailable", model: "claude-sonet-5" }),
        { status: 400 },
      ),
    );
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);

    fireEvent.change(idInput(), { target: { value: "claude-sonet-5" } });
    fireEvent.click(applyBtn());

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent('"claude-sonet-5"');
    expect(alert).toHaveTextContent(/doesn't recognise/i);
    expect(alert.textContent).not.toContain("—");
    expect(onClose).not.toHaveBeenCalled();
    expect(idInput()).toHaveValue("claude-sonet-5");
    expect(applyBtn()).not.toBeDisabled();
  });

  it("says Checking... on the custom path while the bridge checks, Applying... for an alias", async () => {
    let release: (r: Response) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(new Promise<Response>((res) => (release = res))),
    );
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    fireEvent.change(idInput(), { target: { value: "claude-sonnet-5" } });
    fireEvent.click(applyBtn());
    await waitFor(() => expect(screen.getByRole("button", { name: /checking/i })).toBeDisabled());
    release(new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5" }), { status: 200 }));
    cleanup();

    let releaseAlias: (r: Response) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(new Promise<Response>((res) => (releaseAlias = res))),
    );
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    fireEvent.click(screen.getByLabelText(/opus/i));
    fireEvent.click(applyBtn());
    await waitFor(() => expect(screen.getByRole("button", { name: /applying/i })).toBeDisabled());
    releaseAlias(new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }));
  });

  it("the new custom-option strings contain no em-dash", () => {
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    fireEvent.change(idInput(), { target: { value: "Bad ID" } });
    const customRow = customRadio().closest("label")!;
    expect(customRow.textContent).not.toContain("—");
    expect(screen.getByText(/start with claude-/i).textContent).not.toContain("—");
  });

  it("reports an unconfirmed custom id to the caller (saved, but the bridge could not check it)", async () => {
    stubFetch(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: false }), { status: 200 }),
    );
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);
    fireEvent.change(idInput(), { target: { value: "claude-sonnet-5" } });
    fireEvent.click(applyBtn());
    await waitFor(() => expect(onClose).toHaveBeenCalledWith("claude-sonnet-5", { checked: false }));
  });

  it("a confirmed or alias apply calls onClose with the model only", async () => {
    stubFetch(new Response(JSON.stringify({ ok: true, model: "opus" }), { status: 200 }));
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);
    fireEvent.click(screen.getByLabelText(/opus/i));
    fireEvent.click(applyBtn());
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onClose.mock.calls[0]).toEqual(["opus"]);
  });

  it("does not accept an id over 64 characters", () => {
    render(<ModelPicker current="sonnet" persona="test" onClose={() => {}} />);
    fireEvent.change(idInput(), { target: { value: "claude-" + "a".repeat(58) } });
    expect(applyBtn()).toBeDisabled();
  });

  it("passes the bridge's tier fallbacks to the caller", async () => {
    const fb = [{ family: "haiku", wanted: "claude-haiku-5", using: "haiku" }];
    stubFetch(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true, fallbacks: fb }), { status: 200 }),
    );
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);
    fireEvent.change(idInput(), { target: { value: "claude-sonnet-5" } });
    fireEvent.click(applyBtn());
    await waitFor(() => expect(onClose).toHaveBeenCalledWith("claude-sonnet-5", { fallbacks: fb }));
  });

  it("passes the tiers the bridge could not check yet to the caller", async () => {
    stubFetch(
      new Response(JSON.stringify({ ok: true, model: "claude-sonnet-5", checked: true, tiers_unchecked: ["claude-haiku-5"] }), { status: 200 }),
    );
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} />);
    fireEvent.change(idInput(), { target: { value: "claude-sonnet-5" } });
    fireEvent.click(applyBtn());
    await waitFor(() => expect(onClose).toHaveBeenCalledWith("claude-sonnet-5", { tiersUnchecked: ["claude-haiku-5"] }));
  });

  it("allowReapply lets the user apply the model that is already current (an unsupported setting fell back to it)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, model: "sonnet" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();
    render(<ModelPicker current="sonnet" persona="test" onClose={onClose} allowReapply />);
    const apply = screen.getByRole("button", { name: /apply/i });
    expect(apply).not.toBeDisabled();
    fireEvent.click(apply);
    await waitFor(() => {
      expect(onClose).toHaveBeenCalledWith("sonnet");
    });
    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ model: "sonnet" });
  });
});
