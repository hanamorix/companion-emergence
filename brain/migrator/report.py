"""Migration report formatting + source-manifest writer."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.migrator.og import FileManifest
from brain.migrator.transform import SkippedMemory


@dataclass(frozen=True)
class MigrationReport:
    """Aggregated outcome of a single migrator run."""

    memories_migrated: int
    memories_skipped: list[SkippedMemory]
    edges_migrated: int
    edges_skipped: int
    elapsed_seconds: float
    source_manifest: list[FileManifest]
    next_steps_inspect_cmds: list[str]
    next_steps_install_cmd: str
    reflex_arcs_migrated: int = 0
    reflex_arcs_skipped_reason: str | None = None
    interests_migrated: int = 0
    interests_skipped_reason: str | None = None
    vocabulary_emotions_migrated: int = 0
    vocabulary_skipped_reason: str | None = None


def format_report(report: MigrationReport) -> str:
    """Return the human-readable report text (printed + written to migration-report.md)."""
    lines: list[str] = []
    lines.append("Migration complete.")
    lines.append("")
    lines.append(
        f"  Memories:       {report.memories_migrated:,} migrated, "
        f"{len(report.memories_skipped):,} skipped"
    )
    lines.append(
        f"  Hebbian edges:  {report.edges_migrated:,} migrated, {report.edges_skipped:,} skipped"
    )
    lines.append(
        f"  Reflex arcs:    {report.reflex_arcs_migrated:,} migrated"
        + (
            f" (skipped: {report.reflex_arcs_skipped_reason})"
            if report.reflex_arcs_skipped_reason
            else ""
        )
    )
    lines.append(
        f"  Vocabulary:     {report.vocabulary_emotions_migrated:,} emotions migrated"
        + (
            f" (skipped: {report.vocabulary_skipped_reason})"
            if report.vocabulary_skipped_reason
            else ""
        )
    )
    lines.append(
        f"  Interests:      {report.interests_migrated:,} migrated"
        + (
            f" (skipped: {report.interests_skipped_reason})"
            if report.interests_skipped_reason
            else ""
        )
    )
    lines.append(f"  Elapsed:        {report.elapsed_seconds:.1f}s")
    lines.append("")

    if report.memories_skipped:
        lines.append(f"Skipped memories ({len(report.memories_skipped)}):")
        counts = Counter(s.reason for s in report.memories_skipped)
        for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {n} {reason}")
        lines.append("")

    if report.source_manifest:
        lines.append("Source manifest:")
        for m in report.source_manifest:
            lines.append(
                f"  {m.relative_path:<32} {m.size_bytes:>12,} bytes  sha256={m.sha256[:12]}..."
            )
        lines.append("")

    if report.next_steps_inspect_cmds or report.next_steps_install_cmd:
        lines.append("Next steps:")
        step = 1
        if report.next_steps_inspect_cmds:
            lines.append(f"  {step}. Inspect the output:")
            for cmd in report.next_steps_inspect_cmds:
                lines.append(f"       {cmd}")
            step += 1
        if report.next_steps_install_cmd:
            if step > 1:
                lines.append("")
            lines.append(f"  {step}. When satisfied, install as a persona:")
            lines.append(f"       {report.next_steps_install_cmd}")

    return "\n".join(lines) + "\n"


def write_source_manifest(path: Path, manifest: list[FileManifest]) -> None:
    """Write source-manifest.json with every FileManifest entry + a generation timestamp."""
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "files": [
            {
                "relative_path": m.relative_path,
                "size_bytes": m.size_bytes,
                "sha256": m.sha256,
                "mtime_utc": m.mtime_utc,
            }
            for m in manifest
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
