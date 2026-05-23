// Tests for StepInstalling — Bundle 10 (migration summary card)

import { describe, it, expect, afterEach, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { StepInstalling } from "./StepInstalling";

import type { MigrationReport } from "../../appConfig";

afterEach(cleanup);

function mkCEReport(overrides: Partial<MigrationReport> = {}): MigrationReport {
  return {
    kind: "MigrationReport",
    source_kind: "companion-emergence",
    memories_migrated: 0,
    memories_skipped: [],
    edges_migrated: 0,
    crystallizations_migrated: 12,
    bytes_copied: 15_728_640,
    elapsed_seconds: 1.2,
    ...overrides,
  };
}

function mkKitReport(overrides: Partial<MigrationReport> = {}): MigrationReport {
  return {
    kind: "MigrationReport",
    source_kind: "emergence-kit",
    memories_migrated: 47,
    memories_skipped: [
      { id: "mem-001", reason: "missing_content", field: "text", raw_snippet: "" },
      { id: "mem-002", reason: "missing_content", field: "text", raw_snippet: "" },
      { id: "mem-003", reason: "encoding_error", field: "text", raw_snippet: "" },
    ],
    edges_migrated: 0,
    crystallizations_migrated: 5,
    bytes_copied: 0,
    elapsed_seconds: 0.8,
    personality_copied: true,
    ...overrides,
  };
}

function stdoutWith(report: MigrationReport): string {
  return `Running migrator...\nDone.\n${JSON.stringify(report)}`;
}

function renderInstalling(stdout: string, ok = true) {
  return render(
    <StepInstalling
      step={2}
      totalSteps={4}
      result={{ ok, output: stdout, error: "" }}
      onRetry={vi.fn()}
      onBack={vi.fn()}
      avatar={<div data-testid="avatar" />}
    />,
  );
}

describe("StepInstalling — MigrationReport summary card", () => {
  it("fixture 1: companion-emergence report shows bytes and forward-copy label", () => {
    const report = mkCEReport();
    renderInstalling(stdoutWith(report));
    expect(screen.getByText(/15\.0 MB/i)).toBeInTheDocument();
    expect(screen.getByText(/forward.copy/i)).toBeInTheDocument();
    expect(screen.queryByText(/Running migrator/)).not.toBeInTheDocument();
  });

  it("fixture 2: emergence-kit report with skips shows migrated count + skip breakdown", () => {
    const report = mkKitReport();
    renderInstalling(stdoutWith(report));
    expect(screen.getByText(/47 memories migrated/i)).toBeInTheDocument();
    expect(screen.getByText(/missing_content/i)).toBeInTheDocument();
    expect(screen.getByText(/encoding_error/i)).toBeInTheDocument();
  });

  it("fixture 3: plain non-JSON stdout falls back to raw pre", () => {
    const rawOutput = "uv run nell init\nSetup complete.";
    renderInstalling(rawOutput);
    expect(screen.getByText(/Setup complete\./)).toBeInTheDocument();
    expect(screen.queryByText(/forward.copy/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/memories migrated/i)).not.toBeInTheDocument();
  });

  it("stdout ending in non-MigrationReport JSON falls back to raw pre", () => {
    const output = 'some text\n{"kind":"SomethingElse","value":1}';
    renderInstalling(output);
    expect(screen.getByText(/some text/)).toBeInTheDocument();
    expect(screen.queryByText(/forward.copy/i)).not.toBeInTheDocument();
  });

  it("crystallizations count shown for companion-emergence report", () => {
    const report = mkCEReport({ crystallizations_migrated: 12 });
    renderInstalling(stdoutWith(report));
    expect(screen.getByText(/12 crystallization/i)).toBeInTheDocument();
  });

  it("error branch renders raw output+error regardless of report in stdout", () => {
    const report = mkCEReport();
    render(
      <StepInstalling
        step={2}
        totalSteps={4}
        result={{ ok: false, output: stdoutWith(report), error: "Exit code 1" }}
        onRetry={vi.fn()}
        onBack={vi.fn()}
        avatar={<div />}
      />,
    );
    expect(screen.getByText(/Exit code 1/)).toBeInTheDocument();
  });
});
