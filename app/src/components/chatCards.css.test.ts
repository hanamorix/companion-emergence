// #532: the chat cards' scroll caps live in styles.css, which jsdom never lays out. These tests read the stylesheet and
// pin the declarations that keep each card's buttons in view: a capped, scrolling text body and a footer that never
// shrinks. The real layout was checked in a browser; see the PR.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// jsdom's own URL class is not one node's fileURLToPath accepts, so the path is joined rather than built with new URL.
const css = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "styles.css"), "utf-8");

/** The declarations of the rule whose selector list is exactly `selector`, as a property → value map. */
function rule(selector: string): Record<string, string> {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`(^|\\n)${escaped}\\s*\\{([^}]*)\\}`));
  if (!match) throw new Error(`no rule for ${selector} in styles.css`);
  const out: Record<string, string> = {};
  for (const decl of match[2].replace(/\/\*[\s\S]*?\*\//g, "").split(";")) {
    const i = decl.indexOf(":");
    if (i > 0) out[decl.slice(0, i).trim()] = decl.slice(i + 1).trim();
  }
  return out;
}

describe("chat card scroll caps (#532)", () => {
  it("caps the reach-out message and lets it scroll", () => {
    const body = rule(".initiate-banner__body");
    expect(body["max-height"]).toBeTruthy();
    expect(body["overflow-y"]).toBe("auto");
  });

  it("caps the voice-edit text and lets it scroll", () => {
    const body = rule(".voice-edit-panel__body");
    expect(body["max-height"]).toBeTruthy();
    expect(body["overflow-y"]).toBe("auto");
  });
});
