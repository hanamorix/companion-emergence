"""#484: three writers whose comments promise create-only / atomic behaviour that they did not have.

* `commit_note` picked the first name that did not exist and then wrote with "w" (truncate): two notes with the same date and subject
  committed between the check and the write both chose the same name and the later one overwrote the earlier. It now creates the file
  exclusively and moves to the next suffix when the name is taken.
* The systemd unit file and the Windows task XML were written in place, so a crash or a full disk mid-write left a truncated file for
  the next `systemctl` / `schtasks` call. They now go through the shared atomic text writer (unique temp, replace).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

import brain.notes.write as notes_write
from brain.notes.compose import Note
from brain.notes.write import commit_note

NOW = datetime(2026, 6, 15, tzinfo=UTC)


@pytest.fixture
def dirs(tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    return persona, folder


def test_a_note_committed_between_the_name_check_and_the_write_is_not_overwritten(dirs, monkeypatch):
    persona, folder = dirs
    real_guard = notes_write.check_write_target
    rival = folder / "2026-06-15 — dup.md"

    def guard_then_rival_lands(*args, **kwargs):
        result = real_guard(*args, **kwargs)
        if not rival.exists():
            rival.write_text("the rival note", encoding="utf-8")  # a second tick wins the race for the first name
        return result

    monkeypatch.setattr(notes_write, "check_write_target", guard_then_rival_lands)

    path = commit_note(persona, folder, Note(subject="dup", body="mine"), now=NOW)

    assert rival.read_text(encoding="utf-8") == "the rival note"  # untouched
    assert path is not None and Path(path) != rival and "mine" in Path(path).read_text(encoding="utf-8")


def test_collision_suffixes_still_count_up(dirs):
    # PIN
    persona, folder = dirs
    paths = [commit_note(persona, folder, Note(subject="dup", body=str(i)), now=NOW) for i in range(3)]

    assert [Path(p).name for p in paths] == ["2026-06-15 — dup.md", "2026-06-15 — dup-2.md", "2026-06-15 — dup-3.md"]


def test_a_refused_target_still_returns_none(dirs, monkeypatch):
    # PIN
    persona, folder = dirs
    monkeypatch.setattr(notes_write, "is_within_authorized", lambda *a, **k: False)

    assert commit_note(persona, folder, Note(subject="x", body="y"), now=NOW) is None


def _sandbox_windows_paths(monkeypatch, tmp_path):
    """Point every location the Windows backend resolves into tmp_path (as test_windows_service.py does)."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    monkeypatch.setenv("KINDLED_HOME", str(home / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))


def _kill_before_replace(monkeypatch):
    def boom(src, dst, *a, **k):
        raise OSError("killed before the replace")

    monkeypatch.setattr(os, "replace", boom)


def test_the_systemd_unit_survives_a_write_that_dies_before_the_replace(tmp_path, monkeypatch):
    from brain.service import systemd

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    nell = tmp_path / "nell"
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)
    unit = systemd.paths_for_persona("canary").unit_path
    unit.parent.mkdir(parents=True)
    unit.write_text("the previous, complete unit", encoding="utf-8")
    _kill_before_replace(monkeypatch)

    with pytest.raises(OSError):
        systemd.write_systemd_unit(persona="canary", nell_path=nell)

    assert unit.read_text(encoding="utf-8") == "the previous, complete unit"
    assert [p.name for p in unit.parent.iterdir() if ".new" in p.name or p.name.endswith(".tmp")] == []  # no temp left behind


def test_the_systemd_unit_is_still_written_whole(tmp_path, monkeypatch):
    # PIN
    from brain.service import systemd

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    nell = tmp_path / "nell"
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)

    path = systemd.write_systemd_unit(persona="canary", nell_path=nell)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("[Unit]\n") and "--client-origin systemd" in text and text.endswith("WantedBy=default.target\n")


def test_the_windows_task_xml_survives_a_write_that_dies_before_the_replace(tmp_path, monkeypatch):
    from brain.service import windows_service

    _sandbox_windows_paths(monkeypatch, tmp_path)
    nell = tmp_path / "nell.exe"
    nell.write_text("x")
    xml = windows_service.paths_for_persona("canary").xml_path
    xml.parent.mkdir(parents=True)
    xml.write_text("the previous, complete task", encoding="utf-16")
    _kill_before_replace(monkeypatch)

    with pytest.raises(OSError):
        windows_service.write_task_xml(persona="canary", nell_path=nell)

    assert xml.read_text(encoding="utf-16") == "the previous, complete task"
    assert str(tmp_path) in str(xml)  # sandboxed: nothing outside tmp_path was touched


def test_the_windows_task_xml_keeps_its_utf16_encoding(tmp_path, monkeypatch):
    # PIN: the XML declaration says UTF-16, so the file is UTF-16 with a byte-order mark.
    from brain.service import windows_service

    _sandbox_windows_paths(monkeypatch, tmp_path)
    nell = tmp_path / "nell.exe"
    nell.write_text("x")

    path = windows_service.write_task_xml(persona="canary", nell_path=nell)

    raw = path.read_bytes()
    assert raw[:2] in (b"\xff\xfe", b"\xfe\xff") and "<?xml" in raw.decode("utf-16")
