"""#511: the heal's default-text writes go through the shared atomic writer.

`attempt_heal_text` (a missing file) and `_heal_text_from_baks` (every backup empty) wrote the default with a truncating
`path.write_text`: a crash in the middle left a short or empty file. They now call `save_with_backup_text(...,
keep_backups=False)`; `keep_backups=False` matters, or a default write would rotate the .bak chain and destroy a backup.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain.health.attempt_heal import attempt_heal_text


class SimulatedKill(BaseException):
    """Not an Exception: no cleanup runs for it, as for a real kill."""


def _boom_fsync(monkeypatch, exc=None):
    def boom(fd):
        raise exc or OSError(28, "disk full")

    monkeypatch.setattr(os, "fsync", boom)


def _temps(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".new"))


def test_a_missing_file_gets_its_default_atomically(tmp_path):
    p = tmp_path / "voice.md"
    text, anomaly = attempt_heal_text(p, default_factory=lambda: "default voice \u2014 \u00e9\n")
    assert anomaly is None and text == "default voice \u2014 \u00e9\n"
    # Exactly the bytes the old truncating `write_text` produced on THIS platform: text mode, so "\n" is "\r\n" on
    # Windows. The writer keeps that (newline=None, the same open as write_text); forcing "\n" would change Windows.
    reference = tmp_path / "reference.md"
    reference.write_text("default voice \u2014 \u00e9\n", encoding="utf-8")
    assert p.read_bytes() == reference.read_bytes()
    assert _temps(tmp_path) == []


def test_a_failed_default_write_for_a_missing_file_leaves_it_absent_not_short(
    tmp_path, monkeypatch
):
    p = tmp_path / "voice.md"
    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        attempt_heal_text(p, default_factory=lambda: "x" * 5000)
    assert not p.exists()
    assert _temps(tmp_path) == []


def test_a_kill_during_the_default_write_leaves_no_short_file_and_the_next_call_sweeps_the_temp(
    tmp_path, monkeypatch
):
    p = tmp_path / "voice.md"
    _boom_fsync(monkeypatch, SimulatedKill())
    with pytest.raises(SimulatedKill):
        attempt_heal_text(p, default_factory=lambda: "x" * 5000)
    assert not p.exists() and len(_temps(tmp_path)) == 1
    monkeypatch.undo()
    text, _ = attempt_heal_text(p, default_factory=lambda: "fresh")
    assert text == "fresh" and p.read_text(encoding="utf-8") == "fresh" and _temps(tmp_path) == []


def test_a_failed_reset_to_default_after_quarantine_leaves_the_live_path_absent_not_short(
    tmp_path, monkeypatch
):
    p = tmp_path / "voice.md"
    p.write_text(
        "", encoding="utf-8"
    )  # empty live file, no backups: the heal resets to the default
    _boom_fsync(monkeypatch)
    with pytest.raises(OSError):
        attempt_heal_text(p, default_factory=lambda: "x" * 5000)
    assert not p.exists()
    assert (
        len(list(tmp_path.glob("voice.md.corrupt-*"))) == 1
    )  # the damaged file was quarantined, not lost
    assert _temps(tmp_path) == []


def test_the_reset_to_default_writes_the_default_when_nothing_fails(tmp_path):
    p = tmp_path / "voice.md"
    p.write_text("", encoding="utf-8")
    text, anomaly = attempt_heal_text(p, default_factory=lambda: "DEFAULT")
    assert text == "DEFAULT" and anomaly is not None and anomaly.action == "reset_to_default"
    assert p.read_text(encoding="utf-8") == "DEFAULT" and _temps(tmp_path) == []


def test_a_default_write_for_a_missing_file_does_not_rotate_the_backup_chain(tmp_path):
    """keep_backups=False: a missing live file with backups present must not drop or shift a backup."""
    p = tmp_path / "voice.md"
    (tmp_path / "voice.md.bak1").write_text("one", encoding="utf-8")
    (tmp_path / "voice.md.bak2").write_text("two", encoding="utf-8")
    (tmp_path / "voice.md.bak3").write_text("three", encoding="utf-8")
    attempt_heal_text(p, default_factory=lambda: "default")
    assert [(tmp_path / f"voice.md.bak{i}").read_text(encoding="utf-8") for i in (1, 2, 3)] == [
        "one",
        "two",
        "three",
    ]
    assert p.read_text(encoding="utf-8") == "default"
