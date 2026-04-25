"""Tests for brain.user_preferences — GUI-surfaceable cadence file."""

from __future__ import annotations

from pathlib import Path

from brain.user_preferences import (
    DEFAULT_DREAM_EVERY_HOURS,
    UserPreferences,
    read_raw_keys,
)

# ---- Task 9: attempt_heal wiring ----


def test_user_preferences_load_corrupt_file_quarantines_and_resets(tmp_path: Path) -> None:
    """Corrupt JSON → defaults returned + quarantine file present."""
    path = tmp_path / "user_preferences.json"
    path.write_text("{not json at all", encoding="utf-8")

    prefs, anomaly = UserPreferences.load_with_anomaly(path)

    assert prefs.dream_every_hours == DEFAULT_DREAM_EVERY_HOURS
    assert anomaly is not None
    assert anomaly.kind == "json_parse_error"
    corrupt_files = list(tmp_path.glob("user_preferences.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_user_preferences_load_corrupt_file_restores_from_bak(tmp_path: Path) -> None:
    """Valid .bak1 + corrupt live file → .bak1 content returned."""
    path = tmp_path / "user_preferences.json"
    bak1 = tmp_path / "user_preferences.json.bak1"
    bak1.write_text('{"dream_every_hours": 8.0}\n', encoding="utf-8")
    path.write_text("{corrupt", encoding="utf-8")

    prefs, anomaly = UserPreferences.load_with_anomaly(path)

    assert prefs.dream_every_hours == 8.0
    assert anomaly is not None
    assert "bak1" in anomaly.action


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    prefs = UserPreferences.load(tmp_path / "nope.json")
    assert prefs.dream_every_hours == DEFAULT_DREAM_EVERY_HOURS


def test_load_well_formed_file(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text('{"dream_every_hours": 12.0}\n', encoding="utf-8")
    prefs = UserPreferences.load(path)
    assert prefs.dream_every_hours == 12.0


def test_load_corrupt_json_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text("{not json", encoding="utf-8")
    prefs = UserPreferences.load(path)
    assert prefs.dream_every_hours == DEFAULT_DREAM_EVERY_HOURS


def test_load_non_object_payload_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    prefs = UserPreferences.load(path)
    assert prefs.dream_every_hours == DEFAULT_DREAM_EVERY_HOURS


def test_load_wrong_field_type_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text('{"dream_every_hours": "not-a-number"}', encoding="utf-8")
    prefs = UserPreferences.load(path)
    assert prefs.dream_every_hours == DEFAULT_DREAM_EVERY_HOURS


def test_save_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    UserPreferences(dream_every_hours=8.0).save(path)
    assert UserPreferences.load(path).dream_every_hours == 8.0


def test_save_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    UserPreferences(dream_every_hours=12.0).save(path)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".new").exists()


# ---- read_raw_keys ----


def test_read_raw_keys_missing_file(tmp_path: Path) -> None:
    assert read_raw_keys(tmp_path / "missing.json") == set()


def test_read_raw_keys_returns_present_keys(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text('{"dream_every_hours": 8.0, "future_field": 1}', encoding="utf-8")
    assert read_raw_keys(path) == {"dream_every_hours", "future_field"}


def test_read_raw_keys_corrupt_json_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text("not json", encoding="utf-8")
    assert read_raw_keys(path) == set()


def test_read_raw_keys_non_object_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "user_preferences.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_raw_keys(path) == set()
