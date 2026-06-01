/**
 * Tauri-2 arg-naming convention regression gate.
 *
 * Tauri 2's `#[tauri::command]` macro auto-converts Rust snake_case parameter
 * names to camelCase on the JS-side serialisation boundary unless the command
 * is annotated with `#[tauri::command(rename_all = "snake_case")]`. The
 * project's Rust commands use the Tauri default (no rename_all), so DIRECT
 * multi-word args from JS MUST be camelCase.
 *
 * Struct-wrapped args (`invoke("...", { args: { ... } })`) follow a different
 * rule: the inner-struct fields are deserialised by serde from a struct
 * definition whose fields are snake_case by default, so snake_case keys are
 * correct INSIDE the args wrapper.
 *
 * The bug this gate catches: `runPreflightExistingCE` shipped to v0.0.28-alpha.1
 * passing `{ input_dir }` (snake_case) for a direct multi-word arg. Tauri 2
 * rejected it at runtime with: `invalid args 'inputDir' for command
 * 'preflight_existing_ce': command preflight_existing_ce missing required key
 * inputDir`. The unit tests passed because they mock `invoke()` and never go
 * through real deserialisation.
 *
 * This test enumerates every `invoke()` call in app/src/ and asserts the keys
 * follow the convention. New direct multi-word snake_case args will fail here
 * before they ship.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const SRC_ROOT = dirname(fileURLToPath(import.meta.url));

function listSourceFiles(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      if (entry === "node_modules" || entry === "dist" || entry === "src-tauri") continue;
      listSourceFiles(full, acc);
    } else if (
      (entry.endsWith(".ts") || entry.endsWith(".tsx")) &&
      !entry.endsWith(".test.ts") &&
      !entry.endsWith(".test.tsx") &&
      !entry.endsWith(".d.ts")
    ) {
      acc.push(full);
    }
  }
  return acc;
}

interface InvokeCall {
  file: string;
  line: number;
  command: string;
  argsLiteral: string;
}

// Matches `invoke<T>("command_name", { ... })` or `invoke("command_name", { ... })`.
// The args literal is captured up to the matching closing brace (single-line only;
// multi-line invocations are rare and can be linted manually if introduced).
const INVOKE_RE = /\binvoke(?:<[^>]*>)?\s*\(\s*["']([a-z_][a-z0-9_]*)["']\s*,\s*(\{[^}]*\})\s*\)/g;

function findInvokeCalls(file: string): InvokeCall[] {
  const text = readFileSync(file, "utf8");
  const calls: InvokeCall[] = [];
  for (const match of text.matchAll(INVOKE_RE)) {
    const command = match[1];
    const argsLiteral = match[2];
    const line = text.slice(0, match.index ?? 0).split("\n").length;
    calls.push({ file: relative(SRC_ROOT, file), line, command, argsLiteral });
  }
  return calls;
}

// Top-level keys from an args object literal like `{ persona, sourceDir }` or
// `{ args: { persona, source_dir, dry_run }, force }` — we want only the OUTER
// keys (struct wrapping makes inner-struct keys exempt from the convention).
function topLevelKeys(argsLiteral: string): string[] {
  const inner = argsLiteral.slice(1, -1).trim();
  if (!inner) return [];
  const keys: string[] = [];
  let depth = 0;
  let token = "";
  const flush = () => {
    const k = token.trim();
    if (k) {
      // Handle `{ persona }` (shorthand), `{ persona: value }`, `{ "persona": value }`
      const colonIdx = k.indexOf(":");
      const name = (colonIdx >= 0 ? k.slice(0, colonIdx) : k).trim().replace(/^["']|["']$/g, "");
      if (name) keys.push(name);
    }
    token = "";
  };
  for (const ch of inner) {
    if (ch === "{" || ch === "[" || ch === "(") depth++;
    else if (ch === "}" || ch === "]" || ch === ")") depth--;
    else if (ch === "," && depth === 0) {
      flush();
      continue;
    }
    token += ch;
  }
  flush();
  return keys;
}

const SNAKE_CASE_RE = /^[a-z]+(_[a-z0-9]+)+$/;
const STRUCT_WRAPPER_KEYS = new Set(["args"]);

describe("Tauri arg-naming convention", () => {
  const files = listSourceFiles(SRC_ROOT);
  const allCalls = files.flatMap(findInvokeCalls);

  it("finds at least one invoke() call (sanity)", () => {
    expect(allCalls.length).toBeGreaterThan(0);
  });

  it("every direct multi-word invoke() arg uses camelCase, not snake_case", () => {
    const violations: string[] = [];
    for (const call of allCalls) {
      const keys = topLevelKeys(call.argsLiteral);
      for (const key of keys) {
        if (STRUCT_WRAPPER_KEYS.has(key)) continue; // struct-wrapped path is exempt
        if (SNAKE_CASE_RE.test(key)) {
          violations.push(
            `${call.file}:${call.line} invoke("${call.command}", { ${key}: ... }) — ` +
              `direct multi-word arg uses snake_case. Tauri 2 expects camelCase ` +
              `(e.g., ${key.replace(/_([a-z])/g, (_, c) => c.toUpperCase())}). ` +
              `If the Rust command sets rename_all = "snake_case", document it ` +
              `and add the command name to the exemption list in this test.`,
          );
        }
      }
    }
    expect(violations, violations.join("\n\n")).toEqual([]);
  });
});
