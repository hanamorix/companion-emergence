import { describe, it, expect } from "vitest";
import { errString } from "./errString";

describe("errString", () => {
  it("returns Error.message for Error instances", () => {
    expect(errString(new Error("boom"))).toBe("boom");
  });

  it("returns the string as-is for string rejections (Tauri)", () => {
    expect(errString("spawn nell init: ENOENT")).toBe("spawn nell init: ENOENT");
  });

  it("JSON-stringifies plain objects", () => {
    expect(errString({ code: 42, msg: "x" })).toBe('{"code":42,"msg":"x"}');
  });

  it("falls back to String() for unstringifiable objects", () => {
    const cycle: Record<string, unknown> = {};
    cycle.self = cycle;
    expect(errString(cycle)).toBe("[object Object]");
  });

  it("handles null", () => { expect(errString(null)).toBe("null"); });
  it("handles undefined", () => { expect(errString(undefined)).toBe("undefined"); });
  it("handles numbers", () => { expect(errString(42)).toBe("42"); });
  it("handles booleans", () => { expect(errString(false)).toBe("false"); });
});
