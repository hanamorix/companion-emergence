import json

from brain.recovery.report import RecoveryReport, format_report


def test_recovery_report_to_json_roundtrips():
    r = RecoveryReport(
        persona="Phoebe", mode="source", source_dir="/src",
        memories_restored_full=3, memories_restored_summary=0,
        memories_unfaded=1, edges_repaired=4, edges_pruned_unrecoverable=0,
        backup_path="/bak", elapsed_seconds=0.2, dry_run=False,
    )
    payload = json.loads(r.to_json())
    assert payload["kind"] == "RecoveryReport"
    assert payload["memories_restored_full"] == 3
    assert payload["mode"] == "source"


def test_format_report_mentions_persona_and_counts():
    r = RecoveryReport(
        persona="Phoebe", mode="graveyard", source_dir=None,
        memories_restored_full=0, memories_restored_summary=2,
        memories_unfaded=0, edges_repaired=1, edges_pruned_unrecoverable=3,
        backup_path=None, elapsed_seconds=0.1, dry_run=True,
    )
    text = format_report(r)
    assert "Phoebe" in text
    assert "graveyard" in text
