// Vitest smoke tests for bridge.ts — pins the audit-2026-05-07 P1-2
// fix so the persona scoping + per-persona cache can't regress silently.

import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the Tauri invoke so we can drive get_bridge_credentials
// without a real Tauri runtime under jsdom. Two distinct personas →
// two distinct credential payloads, asserted both ways.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: { persona: string }) => {
    if (cmd !== "get_bridge_credentials") throw new Error(`unexpected cmd ${cmd}`);
    if (args.persona === "alice") {
      return { port: 41001, auth_token: "alice-tok" };
    }
    if (args.persona === "bob") {
      return { port: 42002, auth_token: "bob-tok" };
    }
    throw new Error(`unknown persona ${args.persona}`);
  }),
}));

import { invoke } from "@tauri-apps/api/core";
import {
  getBridgeCredentials,
  resetBridgeCredentialCache,
} from "./bridge";

describe("getBridgeCredentials", () => {
  beforeEach(() => {
    resetBridgeCredentialCache();
    vi.clearAllMocks();
  });

  it("invokes get_bridge_credentials with the supplied persona", async () => {
    const creds = await getBridgeCredentials("alice");
    expect(creds.port).toBe(41001);
    expect(creds.authToken).toBe("alice-tok");
    expect(creds.url).toBe("http://127.0.0.1:41001");
    expect(invoke).toHaveBeenCalledWith("get_bridge_credentials", {
      persona: "alice",
    });
  });

  it("caches per-persona — second call same persona doesn't re-invoke", async () => {
    await getBridgeCredentials("alice");
    await getBridgeCredentials("alice");
    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it("does not bleed credentials across personas", async () => {
    const a = await getBridgeCredentials("alice");
    const b = await getBridgeCredentials("bob");
    expect(a.port).toBe(41001);
    expect(b.port).toBe(42002);
    expect(a.authToken).not.toBe(b.authToken);
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it("resetBridgeCredentialCache(persona) only invalidates that persona", async () => {
    await getBridgeCredentials("alice");
    await getBridgeCredentials("bob");
    resetBridgeCredentialCache("alice");
    await getBridgeCredentials("alice"); // re-invokes
    await getBridgeCredentials("bob");   // still cached
    expect(invoke).toHaveBeenCalledTimes(3);
  });
});
