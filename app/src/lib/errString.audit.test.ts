import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "fs";
import { join } from "path";

function* walk(dir: string): Generator<string> {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (
      (p.endsWith(".ts") || p.endsWith(".tsx")) &&
      !p.endsWith(".test.ts") &&
      !p.endsWith(".test.tsx")
    )
      yield p;
  }
}

describe("no (e as Error) regressions", () => {
  it("source tree has no `(e as Error).message` patterns", () => {
    const offenders: string[] = [];
    for (const file of walk("src")) {
      const text = readFileSync(file, "utf-8");
      if (/\(\s*\w+\s+as\s+Error\s*\)/.test(text)) offenders.push(file);
    }
    expect(offenders).toEqual([]);
  });
});
