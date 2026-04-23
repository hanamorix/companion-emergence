"""Tests for brain.migrator.report — migration report + source manifest."""

from __future__ import annotations

import json
from pathlib import Path

from brain.migrator.og import FileManifest
from brain.migrator.report import (
    MigrationReport,
    format_report,
    write_source_manifest,
)
from brain.migrator.transform import SkippedMemory


def test_format_report_totals_section() -> None:
    """Report leads with totals (memories migrated/skipped, edges migrated/skipped)."""
    report = MigrationReport(
        memories_migrated=1128,
        memories_skipped=[
            SkippedMemory(id="x", reason="missing_content", field="content", raw_snippet=""),
        ],
        edges_migrated=8808,
        edges_skipped=0,
        elapsed_seconds=2.3,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "1,128 migrated" in text
    assert "1 skipped" in text
    assert "8,808 migrated" in text
    assert "2.3s" in text


def test_format_report_groups_skips_by_reason() -> None:
    """Skipped memories are grouped + counted by reason."""
    skipped = [
        SkippedMemory(id="a", reason="missing_content", field="content", raw_snippet=""),
        SkippedMemory(id="b", reason="missing_content", field="content", raw_snippet=""),
        SkippedMemory(id="c", reason="non_numeric_emotion", field="emotions", raw_snippet=""),
    ]
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=skipped,
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.1,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "2 missing_content" in text
    assert "1 non_numeric_emotion" in text


def test_format_report_includes_manifest_entries() -> None:
    """Source manifest section lists each file with size + sha256 prefix."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[
            FileManifest(
                relative_path="memories_v2.json",
                size_bytes=123456,
                sha256="abc123" + "0" * 58,
                mtime_utc="2024-01-01T00:00:00Z",
            )
        ],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "memories_v2.json" in text
    assert "123,456" in text
    assert "abc123" in text  # sha prefix visible


def test_format_report_includes_next_steps() -> None:
    """Report shows inspect commands + install command."""
    report = MigrationReport(
        memories_migrated=1,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[
            'sqlite3 out/memories.db "SELECT COUNT(*) FROM memories;"',
        ],
        next_steps_install_cmd="uv run brain migrate --input /og --install-as nell",
    )
    text = format_report(report)
    assert "Next steps" in text
    assert "sqlite3 out/memories.db" in text
    assert "--install-as nell" in text


def test_write_source_manifest_produces_valid_json(tmp_path: Path) -> None:
    """write_source_manifest produces valid JSON with the expected structure."""
    manifest = [
        FileManifest(
            relative_path="memories_v2.json",
            size_bytes=100,
            sha256="a" * 64,
            mtime_utc="2024-01-01T00:00:00Z",
        ),
    ]
    out_path = tmp_path / "source-manifest.json"
    write_source_manifest(out_path, manifest)

    data = json.loads(out_path.read_text())
    assert data["files"][0]["relative_path"] == "memories_v2.json"
    assert data["files"][0]["size_bytes"] == 100
    assert data["files"][0]["sha256"] == "a" * 64
    assert "generated_at_utc" in data
