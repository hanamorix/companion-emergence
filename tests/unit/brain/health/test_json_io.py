"""Tests for brain.health.json_io: the one JSON file reader/writer (#438, #436, #434).

Test names start with the lower-case criterion ID of ``1.5-criteria.md`` (``test_cr3_...``).
Faults are injected at the exact call Windows or a failing disk would raise from; nothing here touches a
persona directory, the network or the bridge.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from brain.health.attempt_heal import save_with_backup_text

POSIX = os.name == "posix"


# --------------------------------------------------------------------------------------------------------------
# Helper additions to attempt_heal.save_with_backup_text (keyword-only; defaults reproduce today's behaviour)
# --------------------------------------------------------------------------------------------------------------


def test_cw4_keep_backups_false_makes_no_backup_and_no_copy(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_with_backup_text(p, "one\n", keep_backups=False)
    save_with_backup_text(p, "two\n", keep_backups=False)
    assert p.read_text() == "two\n"
    assert sorted(f.name for f in tmp_path.iterdir()) == ["x.json"]


def test_cw4_keep_backups_false_leaves_an_existing_chain_alone(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    (tmp_path / "x.json.bak1").write_text("old1\n")
    (tmp_path / "x.json.bak2").write_text("old2\n")
    p.write_text("live\n")
    save_with_backup_text(p, "new\n", keep_backups=False)
    assert p.read_text() == "new\n"
    assert (tmp_path / "x.json.bak1").read_text() == "old1\n"
    assert (tmp_path / "x.json.bak2").read_text() == "old2\n"


def test_cw4_backup_count_zero_keeps_its_legacy_meaning(tmp_path: Path) -> None:
    """The existing suite pins backup_count=0 as 'one backup'; keep_backups=False is the new way to ask for none."""
    p = tmp_path / "x.json"
    save_with_backup_text(p, "one\n", backup_count=0)
    save_with_backup_text(p, "two\n", backup_count=0)
    assert (tmp_path / "x.json.bak1").read_text() == "one\n"


def _count_fsyncs(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def spy(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return calls


def test_cw3_fsync_false_skips_the_file_and_directory_fsync(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "x.json"
    calls = _count_fsyncs(monkeypatch)
    save_with_backup_text(p, "a\n", keep_backups=False, fsync=False)
    assert calls == []
    assert p.read_text() == "a\n"


def test_cw3_fsync_default_syncs_the_file_before_publishing_it(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "x.json"
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def replace(src, dst):
        events.append(f"replace:{Path(dst).name}")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    save_with_backup_text(p, "a\n", keep_backups=False)
    assert events.index("fsync") < events.index("replace:x.json")


@pytest.mark.skipif(not POSIX, reason="file modes are POSIX-only")
def test_cw5_mode_is_applied_at_creation_and_to_every_rotated_backup(
    tmp_path: Path, monkeypatch
) -> None:
    p = tmp_path / "token.json"
    p.write_text("legacy\n")
    p.chmod(0o644)  # a previous, wider-mode live file whose copy becomes .bak1
    old_umask = os.umask(0)
    try:
        created: list[int] = []
        real_open = os.open

        def spy(path, flags, mode=0o777, *a, **k):
            if str(path).endswith(".new") and ".bak1" not in str(path):
                created.append(mode)
            return real_open(path, flags, mode, *a, **k)

        monkeypatch.setattr(os, "open", spy)
        save_with_backup_text(p, "secret\n", 2, mode=0o600)
        save_with_backup_text(p, "secret2\n", 2, mode=0o600)
        monkeypatch.undo()
    finally:
        os.umask(old_umask)
    assert created and all(m == 0o600 for m in created), created
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    for i in (1, 2):
        assert stat.S_IMODE((tmp_path / f"token.json.bak{i}").stat().st_mode) == 0o600


def test_cw10_newline_argument_reaches_the_temp_file(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "x.json"
    seen: list[object] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        if self.name.endswith(".new") and not self.name.endswith(".bak1.new"):
            seen.append(kwargs.get("newline", "<absent>"))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    save_with_backup_text(p, "a\nb\n", keep_backups=False, newline="\n")
    monkeypatch.undo()
    assert seen == ["\n"]
    assert p.read_bytes() == b"a\nb\n"


# --------------------------------------------------------------------------------------------------------------
# Helper addition to jsonl_reader._parse_lines: keyword-only strict (the shared line parser)
# --------------------------------------------------------------------------------------------------------------


def _wrap(data: bytes):
    import io

    return io.TextIOWrapper(
        io.BytesIO(data), encoding="utf-8", errors="surrogateescape", newline="\n"
    )


@pytest.mark.parametrize(
    "data,line",
    [
        (b'{"a": 1}\nnot json\n{"b": 2}\n', 2),
        (b'{"a": 1}\n[1, 2]\n', 2),
        (b'{"a": 1}\n{"b": "caf\xff"}\n', 2),
    ],
)
def test_cj3_parse_lines_strict_raises_with_the_line_number(data: bytes, line: int) -> None:
    from brain.health.jsonl_reader import JsonlLineError, _parse_lines

    seen: list[dict] = []
    with pytest.raises(JsonlLineError) as exc:
        for rec in _parse_lines(_wrap(data), Path("x.jsonl"), strict=True):
            seen.append(rec)
    assert exc.value.line == line
    assert seen == [{"a": 1}]


def test_cj3_parse_lines_default_still_skips(caplog) -> None:
    from brain.health.jsonl_reader import _parse_lines

    data = b'{"a": 1}\nnot json\n[1]\n{"b": "caf\xff"}\n{"c": 3}\n'
    with caplog.at_level(logging.WARNING):
        got = list(_parse_lines(_wrap(data), Path("x.jsonl")))
    assert got == [{"a": 1}, {"c": 3}]
    assert len(caplog.records) == 3
