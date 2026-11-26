import { describe, expect, it } from "vitest";

import {
  friendlyChatError,
  modelRejectionMessage,
  modelUnsupportedMessage,
  modelFallbackMessage,
  modelRejectedNowMessage,
} from "./friendlyChatError";

describe("friendlyChatError", () => {
  it("maps provider_failed to actionable Claude guidance", () => {
    const out = friendlyChatError("provider_failed");
    expect(out).toMatch(/Claude Code is installed/i);
    expect(out).toMatch(/signed in/i);
    expect(out).toMatch(/restart the bridge/i);
  });

  it("maps auth_expired to a re-authorise instruction (#246)", () => {
    const out = friendlyChatError("auth_expired");
    expect(out).toMatch(/login/i);
    expect(out).toMatch(/re-?authori[sz]e/i);
    expect(out).toMatch(/connection panel/i);
  });

  it("passes other errors through unchanged", () => {
    expect(friendlyChatError("Bridge unreachable: Failed to fetch")).toBe(
      "Bridge unreachable: Failed to fetch",
    );
  });

  // #440: the Claude CLI rejected the chat model.
  it("names the rejected model and says what to do (model_unavailable:<id>)", () => {
    const out = friendlyChatError("model_unavailable:claude-sonet-5");
    expect(out).toContain('"claude-sonet-5"');
    expect(out).toMatch(/doesn't recognise/i);
    expect(out).toMatch(/can't use it/i);
    expect(out).toMatch(/Model section of the connection panel/i);
    expect(out).toMatch(/Sonnet, Opus or Haiku/);
    expect(out).not.toMatch(/Claude Code is installed/i);
    expect(out).not.toContain("—");
  });

  it("still gives actionable text without a model name and never prints undefined", () => {
    const out = friendlyChatError("model_unavailable");
    expect(out).toMatch(/the chosen model/);
    expect(out).toMatch(/connection panel/i);
    expect(out).not.toMatch(/undefined/);
  });

  it("recognises the HTTP /chat 502 form", () => {
    const out = friendlyChatError('/chat 502: {"detail":"model_unavailable:claude-x"}');
    expect(out).toContain('"claude-x"');
  });

  it("shortens a very long id for display", () => {
    const out = friendlyChatError("model_unavailable:claude-" + "x".repeat(200));
    expect(out.length).toBeLessThan(300);
    expect(out).toContain("...");
  });

  it("leaves auth_expired and provider_failed untouched", () => {
    expect(friendlyChatError("auth_expired")).toMatch(/login has expired/i);
    expect(friendlyChatError("provider_failed")).toMatch(/Claude Code is installed/i);
  });

  it("uses the picker wording when a just-applied model is refused", () => {
    const out = modelRejectionMessage("claude-sonet-5", "picker");
    expect(out).toContain('"claude-sonet-5"');
    expect(out).toMatch(/Check the model ID, or pick Sonnet, Opus or Haiku/);
    expect(out).not.toMatch(/connection panel/i);
    expect(out).not.toContain("—");
  });

  it("words a job that fell back to its default plainly, without an em-dash", () => {
    const out = modelFallbackMessage({ job: "background-classifier", configured: "claude-haiku-5", using: "haiku" });
    expect(out).toBe(
      'The background classifier job is using "haiku" instead of "claude-haiku-5", which Claude rejected.',
    );
    expect(out).not.toContain("—");
  });

  it("says the jobs now run the default model when a configured id was rejected", () => {
    const chat = modelRejectedNowMessage("claude-sonet-5", true, true);
    expect(chat).toBe(
      'Claude rejected the model "claude-sonet-5". It may not exist, or your account may not have access to it. Chat now runs the default model instead. Use change to pick another model.',
    );
    const bg = modelRejectedNowMessage("claude-haiku-5", false, true);
    expect(bg).toContain("Some background jobs now run the default model instead.");
    expect(chat).not.toContain("—");
    expect(bg).not.toContain("—");
  });

  it("says the job keeps failing when the rejected id is the default itself", () => {
    const out = modelRejectedNowMessage("claude-sonnet-5", true, false);
    expect(out).toBe(
      'Claude rejected the model "claude-sonnet-5", which is the default model. Chat keeps failing until Claude Code is updated or you pick another model with change.',
    );
    expect(out).not.toContain("—");
    expect(modelRejectedNowMessage("claude-sonnet-5", false, false)).toContain(
      "keep failing until Claude Code is updated or the model setting is changed.",
    );
    expect(modelRejectedNowMessage("claude-sonnet-5", true, false)).not.toContain("default model instead");
  });

  it("words the unsupported-model setting plainly, naming the value and Sonnet, without an em-dash", () => {
    const out = modelUnsupportedMessage("fable");
    expect(out).toBe(
      'The model "fable" in your settings is not supported, so Sonnet is being used instead. Use change to pick a model and replace it.',
    );
    expect(out).not.toContain("—");
    expect(out).not.toContain("undefined");
  });
});
