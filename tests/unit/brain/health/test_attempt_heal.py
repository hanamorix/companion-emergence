"""Tests for brain.health.attempt_heal — core heal + save helpers."""

from __future__ import annotations

import json
from pathlib import Path

from brain.health.attempt_heal import attempt_heal, save_with_backup


def _default() -> dict:
    return {"version": 1, "value": "default"}


def _vocab_validator(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        raise ValueError("missing 'emotions' list")


# ---- Healthy paths ----


def test_attempt_heal_missing_file_returns_default(tmp_path: Path) -> None:
    data, anomaly = attempt_heal(tmp_path / "x.json", _default)
    assert data == {"version": 1, "value": "default"}
    assert anomaly is None


def test_attempt_heal_well_formed_returns_data(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"version": 1, "value": "stored"}), encoding="utf-8")
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "stored"}
    assert anomaly is None


def test_attempt_heal_validator_passes_well_formed(tmp_path: Path) -> None:
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")
    data, anomaly = attempt_heal(p, _default, schema_validator=_vocab_validator)
    assert anomaly is None


# ---- Corrupt paths ----


def test_attempt_heal_corrupt_json_no_baks_resets_to_default(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "default"}
    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    assert anomaly.kind == "json_parse_error"
    # Quarantine present
    quarantines = list(tmp_path.glob("x.json.corrupt-*"))
    assert len(quarantines) == 1
    # Live file rewritten with default
    assert json.loads(p.read_text(encoding="utf-8")) == {"version": 1, "value": "default"}


def test_attempt_heal_restores_from_bak1(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    bak1 = tmp_path / "x.json.bak1"
    bak1.write_text(json.dumps({"version": 1, "value": "good"}), encoding="utf-8")
    p.write_text("{not json", encoding="utf-8")
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "good"}
    assert anomaly.action == "restored_from_bak1"
    # bak1 is now the live file (was renamed in)
    assert json.loads(p.read_text(encoding="utf-8"))["value"] == "good"
    assert not bak1.exists()  # consumed


def test_attempt_heal_walks_to_bak2_when_bak1_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    (tmp_path / "x.json.bak1").write_text("{also not json", encoding="utf-8")
    (tmp_path / "x.json.bak2").write_text(
        json.dumps({"version": 1, "value": "older_good"}), encoding="utf-8"
    )
    data, anomaly = attempt_heal(p, _default)
    assert data == {"version": 1, "value": "older_good"}
    assert anomaly.action == "restored_from_bak2"


def test_attempt_heal_schema_validator_failure_treated_as_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "vocab.json"
    p.write_text(
        json.dumps({"version": 1, "wrong_field": []}), encoding="utf-8"
    )  # parses but invalid
    data, anomaly = attempt_heal(
        p, lambda: {"version": 1, "emotions": []}, schema_validator=_vocab_validator
    )
    assert anomaly is not None
    assert anomaly.kind == "schema_mismatch"
    assert anomaly.action == "reset_to_default"


def test_attempt_heal_user_edit_heuristic(tmp_path: Path) -> None:
    p = tmp_path / "user_preferences.json"
    p.write_text("{not json", encoding="utf-8")
    # mtime is now (recently edited), small file, content starts with {
    data, anomaly = attempt_heal(p, _default)
    assert anomaly is not None
    assert anomaly.likely_cause == "user_edit"


# ---- Save flow ----


def test_save_with_backup_first_save_no_bak(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_with_backup(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "x.json.bak1").exists()
    assert not (tmp_path / "x.json.new").exists()


def test_save_with_backup_rotates_3_levels(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_with_backup(p, {"a": 1})  # live=1
    save_with_backup(p, {"a": 2})  # live=2, bak1=1
    save_with_backup(p, {"a": 3})  # live=3, bak1=2, bak2=1
    save_with_backup(p, {"a": 4})  # live=4, bak1=3, bak2=2, bak3=1
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 4}
    assert json.loads((tmp_path / "x.json.bak1").read_text(encoding="utf-8")) == {"a": 3}
    assert json.loads((tmp_path / "x.json.bak2").read_text(encoding="utf-8")) == {"a": 2}
    assert json.loads((tmp_path / "x.json.bak3").read_text(encoding="utf-8")) == {"a": 1}


def test_save_with_backup_caps_at_3_drops_oldest(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    for i in range(1, 6):
        save_with_backup(p, {"a": i})
    # live=5, bak1=4, bak2=3, bak3=2, oldest dropped
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 5}
    assert json.loads((tmp_path / "x.json.bak3").read_text(encoding="utf-8")) == {"a": 2}
    assert not (tmp_path / "x.json.bak4").exists()


def test_save_with_backup_unlinks_stale_new(tmp_path: Path) -> None:
    """If .new exists from a prior crash, save unlinks it before writing."""
    p = tmp_path / "x.json"
    (tmp_path / "x.json.new").write_text("stale partial content", encoding="utf-8")
    save_with_backup(p, {"a": 1})
    # Stale .new is gone; live file is correct
    assert not (tmp_path / "x.json.new").exists()
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def test_save_with_backup_higher_count_keeps_more(tmp_path: Path) -> None:
    """When backup_count=6, six backups are retained."""
    p = tmp_path / "x.json"
    for i in range(1, 8):
        save_with_backup(p, {"a": i}, backup_count=6)
    # live=7; bak1..bak6 = 6,5,4,3,2,1
    for k, expected in zip(range(1, 7), [6, 5, 4, 3, 2, 1], strict=True):
        bak = tmp_path / f"x.json.bak{k}"
        assert json.loads(bak.read_text(encoding="utf-8")) == {"a": expected}


def test_quarantine_filename_has_no_colons(tmp_path: Path) -> None:
    """Quarantine filenames must use Windows-safe characters.

    `os.replace` with a filename containing `:` raises WinError 123 on Windows.
    The quarantine timestamp swaps colons for hyphens to round-trip across
    POSIX + Windows.
    """
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    _, anomaly = attempt_heal(p, _default)
    assert anomaly is not None
    assert ":" not in anomaly.quarantine_path
    # Verify the quarantine file actually exists on disk (would have failed
    # to create on Windows if the filename were illegal).
    quarantines = list(tmp_path.glob("x.json.corrupt-*"))
    assert len(quarantines) == 1
    assert ":" not in quarantines[0].name
