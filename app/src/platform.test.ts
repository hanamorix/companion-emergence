import { describe, expect, it } from "vitest";
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
