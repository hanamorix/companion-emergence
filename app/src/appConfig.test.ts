/**
 * appConfig.ts — brain-update invoke wrappers (#286 slice 4).
 *
 * Asserts each wrapper calls invoke() with the exact Rust command name and
 * args, and returns whatever invoke() resolves to.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

import { invoke } from "@tauri-apps/api/core";
import { applyBrainUpdate, checkBrainUpdate, revertBrain, rollbackBrain } from "./appConfig";

const mockInvoke = vi.mocked(invoke);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("checkBrainUpdate", () => {
  it('calls invoke("check_brain_update") and returns its value', async () => {
    const payload = {
      available: true,
      commit: "abc123",
      brain_version: "0.0.44",
      reason: "available",
      detail: null,
    };
    mockInvoke.mockResolvedValueOnce(payload);

    const result = await checkBrainUpdate();

    expect(mockInvoke).toHaveBeenCalledWith("check_brain_update");
    expect(result).toEqual(payload);
  });
});

describe("applyBrainUpdate", () => {
  it('calls invoke("apply_brain_update") and returns its value', async () => {
    const payload = { commit: "abc123", brain_version: "0.0.44" };
    mockInvoke.mockResolvedValueOnce(payload);

    const result = await applyBrainUpdate();

    expect(mockInvoke).toHaveBeenCalledWith("apply_brain_update");
    expect(result).toEqual(payload);
  });
});

describe("rollbackBrain", () => {
  it('calls invoke("rollback_brain", { reason: "why" })', async () => {
    mockInvoke.mockResolvedValueOnce(undefined);

    await rollbackBrain("why");

    expect(mockInvoke).toHaveBeenCalledWith("rollback_brain", { reason: "why" });
  });
});

describe("revertBrain", () => {
  it('calls invoke("revert_brain")', async () => {
    mockInvoke.mockResolvedValueOnce(undefined);

    await revertBrain();

    expect(mockInvoke).toHaveBeenCalledWith("revert_brain");
  });
});

describe("ensureBridgeRunning", () => {
  it("drops the persona's cached bridge credentials, since a spawned bridge writes new ones (#365)", async () => {
    const { getBridgeCredentials, resetBridgeCredentialCache } = await import("./bridge");
    const { ensureBridgeRunning } = await import("./appConfig");
    resetBridgeCredentialCache();
    mockInvoke.mockImplementation(async (cmd) =>
      cmd === "get_bridge_credentials" ? { port: 41001, auth_token: "tok" } : undefined,
    );

    await getBridgeCredentials("alice");
    await ensureBridgeRunning("alice");
    await getBridgeCredentials("alice");

    const credentialReads = mockInvoke.mock.calls.filter(([cmd]) => cmd === "get_bridge_credentials");
    expect(credentialReads).toHaveLength(2);
  });
});
