/**
 * Type-binding smoke tests for appConfig.ts — Bundle 7 of v0.0.18.
 *
 * Covers:
 *   - MigrateSource includes "companion-emergence"
 *   - listPersonas() returns PersonaSummary[] (not string[])
 *   - runPreflightExistingCE passes input_dir as snake_case key
 *   - revealInFileManager swallows errors
 *   - nellbrainHomePath returns null on failure
 *   - MigrationReport and SkippedMemory shapes compile and hold correct values
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

import { invoke } from "@tauri-apps/api/core";
import type {
  ExistingCePreflight,
  MigrationReport,
  MigrateSource,
  PersonaSummary,
  PreflightIssue,
  SkippedMemory,
} from "./appConfig";
import {
  listPersonas,
  nellbrainHomePath,
  revealInFileManager,
  runPreflightExistingCE,
} from "./appConfig";

const mockInvoke = vi.mocked(invoke);

beforeEach(() => {
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// MigrateSource
// ---------------------------------------------------------------------------
describe("MigrateSource", () => {
  it("accepts all three source kinds at compile time", () => {
    const sources: MigrateSource[] = ["nellbrain", "emergence-kit", "companion-emergence"];
    expect(sources).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// listPersonas — PersonaSummary shape
// ---------------------------------------------------------------------------
describe("listPersonas", () => {
  it("returns objects with name and has_memories_db fields (PersonaSummary shape)", async () => {
    const payload: PersonaSummary[] = [
      { name: "nell", last_opened_at: "2026-05-23T10:00:00Z", has_memories_db: true },
    ];
    mockInvoke.mockResolvedValueOnce(payload);
    const result = await listPersonas();
    expect(result[0].name).toBe("nell");
    expect(result[0].has_memories_db).toBe(true);
    expect(result[0].last_opened_at).toBe("2026-05-23T10:00:00Z");
  });

  it("swallows invoke errors and returns empty array", async () => {
    mockInvoke.mockRejectedValueOnce(new Error("bridge down"));
    const result = await listPersonas();
    expect(result).toEqual([]);
  });

  it("handles null last_opened_at", async () => {
    const payload: PersonaSummary[] = [
      { name: "aria", last_opened_at: null, has_memories_db: false },
    ];
    mockInvoke.mockResolvedValueOnce(payload);
    const [first] = await listPersonas();
    expect(first.last_opened_at).toBeNull();
    expect(first.has_memories_db).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// runPreflightExistingCE
// ---------------------------------------------------------------------------
describe("runPreflightExistingCE", () => {
  it("invokes preflight_existing_ce with snake_case input_dir key", async () => {
    const preflight: ExistingCePreflight = {
      ok: true,
      persona_name: "nell",
      imported_user_name: "Hana",
      imported_voice_template: "nell-example",
      memory_count: 42,
      crystallization_count: 7,
      hebbian_edge_count: 200,
      source_size_bytes: 1_500_000,
      errors: [],
      warnings: [],
    };
    mockInvoke.mockResolvedValueOnce(preflight);

    const result = await runPreflightExistingCE("/some/path");
    expect(mockInvoke).toHaveBeenCalledWith("preflight_existing_ce", { input_dir: "/some/path" });
    expect(result.ok).toBe(true);
    expect(result.persona_name).toBe("nell");
    expect(result.memory_count).toBe(42);
  });

  it("propagates errors (caller must handle — no swallow)", async () => {
    mockInvoke.mockRejectedValueOnce(new Error("rust error"));
    await expect(runPreflightExistingCE("/bad")).rejects.toThrow("rust error");
  });

  it("accepts PreflightIssue entries in errors and warnings arrays", async () => {
    const issue: PreflightIssue = { code: "MISSING_DB", message: "No memories.db found" };
    const preflight: ExistingCePreflight = {
      ok: false,
      persona_name: null,
      imported_user_name: null,
      imported_voice_template: null,
      memory_count: null,
      crystallization_count: null,
      hebbian_edge_count: null,
      source_size_bytes: 0,
      errors: [issue],
      warnings: [],
    };
    mockInvoke.mockResolvedValueOnce(preflight);
    const result = await runPreflightExistingCE("/empty");
    expect(result.errors[0].code).toBe("MISSING_DB");
  });
});

// ---------------------------------------------------------------------------
// revealInFileManager
// ---------------------------------------------------------------------------
describe("revealInFileManager", () => {
  it("invokes reveal_in_file_manager with path key", async () => {
    mockInvoke.mockResolvedValueOnce(undefined);
    await revealInFileManager("$HOME");
    expect(mockInvoke).toHaveBeenCalledWith("reveal_in_file_manager", { path: "$HOME" });
  });

  it("swallows errors silently", async () => {
    mockInvoke.mockRejectedValueOnce(new Error("file manager not found"));
    await expect(revealInFileManager("/bad")).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// nellbrainHomePath
// ---------------------------------------------------------------------------
describe("nellbrainHomePath", () => {
  it("returns the home path string on success", async () => {
    mockInvoke.mockResolvedValueOnce("$HOME/.kindled");
    const result = await nellbrainHomePath();
    expect(result).toBe("$HOME/.kindled");
    expect(mockInvoke).toHaveBeenCalledWith("nellbrain_home_path");
  });

  it("returns null and swallows error on failure", async () => {
    mockInvoke.mockRejectedValueOnce(new Error("not available"));
    const result = await nellbrainHomePath();
    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// MigrationReport shape
// ---------------------------------------------------------------------------
describe("MigrationReport shape", () => {
  it("accepts a full MigrationReport including optional personality_copied", () => {
    const skipped: SkippedMemory = {
      id: "mem-123",
      reason: "invalid utf-8",
      field: "content",
      raw_snippet: "???",
    };
    const report: MigrationReport = {
      kind: "MigrationReport",
      source_kind: "companion-emergence",
      memories_migrated: 40,
      memories_skipped: [skipped],
      edges_migrated: 180,
      crystallizations_migrated: 6,
      bytes_copied: 1_400_000,
      elapsed_seconds: 2.3,
      personality_copied: true,
    };
    expect(report.kind).toBe("MigrationReport");
    expect(report.source_kind).toBe("companion-emergence");
    expect(report.memories_skipped[0].id).toBe("mem-123");
  });

  it("allows personality_copied to be absent (optional field)", () => {
    const report: MigrationReport = {
      kind: "MigrationReport",
      source_kind: "nellbrain",
      memories_migrated: 0,
      memories_skipped: [],
      edges_migrated: 0,
      crystallizations_migrated: 0,
      bytes_copied: 0,
      elapsed_seconds: 0.1,
    };
    expect(report.personality_copied).toBeUndefined();
  });
});
