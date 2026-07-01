import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { BrainLoginPrompt } from "./BrainLoginPrompt";
import * as appConfig from "../../appConfig";

vi.mock("../../appConfig");

describe("BrainLoginPrompt", () => {
  beforeEach(() => vi.resetAllMocks());
  afterEach(() => cleanup());

  it("runs the flow: authorize -> shows code field -> submit -> onAuthorized", async () => {
    vi.mocked(appConfig.startBrainLogin).mockResolvedValue({ url: "https://claude.com/x" });
    vi.mocked(appConfig.submitBrainLoginCode).mockResolvedValue({ ok: true, error: null });
    const onAuthorized = vi.fn();
    render(<BrainLoginPrompt onAuthorized={onAuthorized} onDismiss={() => {}} />);

    fireEvent.click(screen.getByRole("button", { name: /authorize/i }));
    await screen.findByLabelText(/paste.*code/i);
    fireEvent.change(screen.getByLabelText(/paste.*code/i), { target: { value: "CODE123" } });
    fireEvent.click(screen.getByRole("button", { name: /finish|submit|verify/i }));
    await waitFor(() => expect(onAuthorized).toHaveBeenCalled());
    expect(appConfig.submitBrainLoginCode).toHaveBeenCalledWith("CODE123");
  });

  it("shows an error and allows retry when the code is rejected", async () => {
    vi.mocked(appConfig.startBrainLogin).mockResolvedValue({ url: "https://claude.com/x" });
    vi.mocked(appConfig.submitBrainLoginCode).mockResolvedValue({ ok: false, error: "code rejected" });
    render(<BrainLoginPrompt onAuthorized={() => {}} onDismiss={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /authorize/i }));
    await screen.findByLabelText(/paste.*code/i);
    fireEvent.change(screen.getByLabelText(/paste.*code/i), { target: { value: "bad" } });
    fireEvent.click(screen.getByRole("button", { name: /finish|submit|verify/i }));
    expect(await screen.findByText(/code rejected/i)).toBeInTheDocument();
  });

  it("calls onDismiss and cancels the login on 'Not now'", async () => {
    const onDismiss = vi.fn();
    render(<BrainLoginPrompt onAuthorized={() => {}} onDismiss={onDismiss} />);
    fireEvent.click(screen.getByRole("button", { name: /not now/i }));
    await waitFor(() => expect(onDismiss).toHaveBeenCalled());
    expect(appConfig.cancelBrainLogin).toHaveBeenCalled();
  });
});
