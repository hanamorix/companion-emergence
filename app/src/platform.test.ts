import { describe, expect, it, vi } from "vitest";
import { platformLabel, supportsMacOnlyInstallActions, type ClientPlatform } from "./platform";

describe("platform helpers", () => {
  it.each([
    ["macos", true, "macOS"],
    ["windows", false, "Windows"],
    ["linux", false, "Linux"],
    ["other", false, "this platform"],
  ] as Array<[ClientPlatform, boolean, string]>) (
    "%s support/label",
    (platform, supported, label) => {
      expect(supportsMacOnlyInstallActions(platform)).toBe(supported);
      expect(platformLabel(platform)).toBe(label);
    },
  );
});

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (_cmd: string) => "appimage"),
}));

import { invoke } from "@tauri-apps/api/core";
import { detectInstallShape } from "./platform";

describe("detectInstallShape", () => {
  it("calls detect_install_shape and returns the string", async () => {
    vi.mocked(invoke).mockResolvedValueOnce("appimage");
    const shape = await detectInstallShape();
    expect(shape).toBe("appimage");
    expect(invoke).toHaveBeenCalledWith("detect_install_shape");
  });

  it("returns 'unknown' when invoke throws", async () => {
    vi.mocked(invoke).mockRejectedValueOnce(new Error("tauri unavailable"));
    const shape = await detectInstallShape();
    expect(shape).toBe("unknown");
  });
});
