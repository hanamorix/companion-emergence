"""Tests for MigrationReport.to_json() + bytes_copied/source_kind fields."""

from __future__ import annotations

import json

from brain.migrator.report import MigrationReport
from brain.migrator.transform import SkippedMemory


def _sample_report(source_kind: str) -> MigrationReport:
    return MigrationReport(
        memories_migrated=412,
        memories_skipped=[
            SkippedMemory(id="abc", reason="missing_content", field="content", raw_snippet="..."),
        ],
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=0.42,
        source_manifest=[],
        next_steps_inspect_cmds=[],
        next_steps_install_cmd="",
        crystallizations_migrated=38,
        crystallizations_skipped_reason=None,
        creative_dna_migrated=False,
        creative_dna_skipped_reason="n/a",
        legacy_files_preserved=1,
        legacy_files_missing=0,
        legacy_skipped_reason=None,
        bytes_copied=12_300_000,
        source_kind=source_kind,
    )


def test_to_json_companion_emergence_report():
    report = _sample_report("companion-emergence")
    payload = json.loads(report.to_json())
    assert payload["kind"] == "MigrationReport"
    assert payload["source_kind"] == "companion-emergence"
    assert payload["memories_migrated"] == 412
    assert payload["bytes_copied"] == 12_300_000
    assert payload["memories_skipped"] == [
        {"id": "abc", "reason": "missing_content", "field": "content", "raw_snippet": "..."},
    ]


def test_to_json_emergence_kit_report_carries_source_kind():
    report = _sample_report("emergence-kit")
    payload = json.loads(report.to_json())
    assert payload["source_kind"] == "emergence-kit"
    assert payload["bytes_copied"] == 12_300_000  # field present even when not meaningful
