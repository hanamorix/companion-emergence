export type ClientPlatform = "macos" | "windows" | "linux" | "other";

export function getClientPlatform(): ClientPlatform {
  const nav = typeof navigator === "undefined" ? null : navigator;
  const platform = (nav?.platform ?? "").toLowerCase();
  const ua = (nav?.userAgent ?? "").toLowerCase();
  const value = `${platform} ${ua}`;

  if (value.includes("mac")) return "macos";
  if (value.includes("win")) return "windows";
  if (value.includes("linux") || value.includes("x11")) return "linux";
  return "other";
}

export function platformLabel(platform: ClientPlatform): string {
  switch (platform) {
    case "macos":
      return "macOS";
    case "windows":
      return "Windows";
    case "linux":
      return "Linux";
    default:
      return "this platform";
  }
}

export function supportsMacOnlyInstallActions(platform: ClientPlatform = getClientPlatform()): boolean {
  return platform === "macos";
}
