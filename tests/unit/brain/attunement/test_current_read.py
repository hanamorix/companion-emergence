"""Tests for current_read.json read/write in store.py."""
from __future__ import annotations

from pathlib import Path

from brain.attunement.schemas import SCHEMA_VERSION, CurrentRead
from brain.attunement.store import read_current_read, write_current_read


def _make_read(ts: str = "2026-05-31T12:00:00Z") -> CurrentRead:
    return CurrentRead(
        ts=ts,
        source_turn_id="turn-001",
        tone_label="warm",
        tone_justification="soft phrasing",
        cadence_label="measured",
        cadence_justification="full sentences",
        mood_valence=0.4,
        mood_intensity=0.5,
        predicted_arc_shape="settling in",
        schema_version=SCHEMA_VERSION,
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    read = _make_read()
    write_current_read(tmp_path, read)
    loaded = read_current_read(tmp_path)
    assert loaded == read


def test_read_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert read_current_read(tmp_path) is None


def test_read_returns_none_when_file_corrupt(tmp_path: Path) -> None:
    (tmp_path / "attunement").mkdir()
    (tmp_path / "attunement" / "current_read.json").write_text("not valid json {")
    assert read_current_read(tmp_path) is None


def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    first = _make_read("2026-05-31T10:00:00Z")
    second = _make_read("2026-05-31T11:00:00Z")
    write_current_read(tmp_path, first)
    write_current_read(tmp_path, second)
    loaded = read_current_read(tmp_path)
    assert loaded == second


def test_write_creates_attunement_dir_if_missing(tmp_path: Path) -> None:
    write_current_read(tmp_path, _make_read())
    assert (tmp_path / "attunement" / "current_read.json").exists()
