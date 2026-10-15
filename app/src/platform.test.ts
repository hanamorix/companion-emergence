import { describe, expect, it, vi } from "vitest";
import { getClientPlatform, platformLabel, supportsMacOnlyInstallActions, supportsNellCliShortcut, type ClientPlatform } from "./platform";

describe("nell shortcut support (#408)", () => {
  it.each([
    ["macos", true],
    ["windows", true],
    ["linux", true],
    ["other", false],
  ] as Array<[ClientPlatform, boolean]>)("%s", (platform, supported) => {
    expect(supportsNellCliShortcut(platform)).toBe(supported);
  });
});

describe("getClientPlatform with real webview user agents (no module mock)", () => {
  const cases: Array<[string, string, string, ClientPlatform]> = [
    ["WebKitGTK on Linux", "Linux x86_64", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15", "linux"],
    ["WebView2 on Windows", "Win32", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0", "windows"],
    ["WKWebView on macOS", "MacIntel", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)", "macos"],
  ];
  it.each(cases)("%s", (_label, platform, userAgent, expected) => {
    vi.stubGlobal("navigator", { platform, userAgent });
    try {
      expect(getClientPlatform()).toBe(expected);
      expect(supportsNellCliShortcut()).toBe(true);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

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

  it("returns 'unknown' when invoke returns an unrecognised string", async () => {
    vi.mocked(invoke).mockResolvedValueOnce("definitely-not-a-shape");
    const shape = await detectInstallShape();
    expect(shape).toBe("unknown");
  });
});
