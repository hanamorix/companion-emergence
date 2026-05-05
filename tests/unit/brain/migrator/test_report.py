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


def test_format_report_renumbers_next_steps_when_inspect_cmds_absent() -> None:
    """When only install_cmd is set, it renders as step 1 — not 'step 2 without 1'.

    The --install-as flow omits inspect cmds (already installed). The report
    must still read coherently rather than leaving a dangling step 2.
    """
    report = MigrationReport(
        memories_migrated=1,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="uv run brain ...",
    )
    text = format_report(report)
    assert "1. When satisfied, install as a persona:" in text
    assert "2. When satisfied" not in text


def test_format_report_empty_report_does_not_crash() -> None:
    """Zero memories + zero edges + no manifest + no next steps produces
    minimal coherent output without raising.
    """
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
    )
    text = format_report(report)
    assert "Migration complete." in text
    assert "0 migrated" in text
    assert "Next steps" not in text  # section suppressed when empty


def test_format_report_shows_creative_dna_and_journal_lines() -> None:
    """Migrated creative_dna + retagged journal memories render in the report."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        creative_dna_migrated=True,
        journal_memories_retagged=14,
    )
    text = format_report(report)
    assert "Creative DNA:" in text
    assert "migrated" in text
    assert "Journal:" in text
    assert "14 memories retagged" in text


def test_format_report_shows_creative_dna_skipped_reason_when_og_missing() -> None:
    """When OG lacks nell_creative_dna.json, the report renders the skipped reason."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        creative_dna_migrated=False,
        creative_dna_skipped_reason="og file not present",
    )
    text = format_report(report)
    assert "Creative DNA:" in text
    assert "not migrated" in text
    assert "skipped: og file not present" in text


def test_format_report_shows_legacy_line() -> None:
    """Migrated + missing legacy file counts render in the report."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        legacy_files_preserved=14,
        legacy_files_missing=2,
    )
    text = format_report(report)
    assert "Legacy files:" in text
    assert "14 preserved" in text
    assert "2 missing" in text


def test_format_report_shows_soul_candidates_line() -> None:
    """soul_candidates_migrated + skipped_missing_memory_id render in the report."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        soul_candidates_migrated=38,
        soul_candidates_skipped_missing_memory_id=2,
    )
    text = format_report(report)
    assert "Soul candidates:" in text
    assert "38 migrated" in text
    assert "2 skipped" in text
    assert "missing memory_id" in text


def test_format_report_shows_reflex_fires_line() -> None:
    """reflex_log_fires_migrated renders in the report."""
    report = MigrationReport(
        memories_migrated=0,
        memories_skipped=[],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.0,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        reflex_log_fires_migrated=42,
    )
    text = format_report(report)
    assert "Reflex fires:" in text
    assert "42 migrated" in text
