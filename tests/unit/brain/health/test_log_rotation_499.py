"""#499: a failed rolling rotation must not lose a retained archive or a log row.

The old order shifted the archives up (evicting the oldest) before it wrote the new `.1.gz`, so while the write kept
failing each tick destroyed one archive. The rotation now writes the new archive to `<log>.next.gz` first, shifts
with a gap-aware `os.replace` chain, publishes with the retried replace and truncates last. Faults are injected at
`os.replace` (the one call every step but the compression goes through) and at `gzip.open`, and a `BaseException`
subclass stands in for a kill (no cleanup may run for it). The tests do not import the shared writer module.
"""

from __future__ import annotations

import errno
import gzip
import os
import re
from pathlib import Path

import pytest

import brain.health.log_rotation as lr

CAP = 100


class SimulatedKill(BaseException):
    pass


def _gz_bytes(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def _make_log(directory: Path, name: str = "a.jsonl", rows: int = 40, tag: str = "L") -> Path:
    log = directory / name
    log.write_bytes("".join(f"{tag}-{i} " + "x" * 20 + "\n" for i in range(rows)).encode())
    return log


def _make_archive(log: Path, n: int, content: str) -> None:
    with gzip.open(lr._archive_path(log, n), "wb") as g:
        g.write(content.encode())


def _setup(directory: Path, keep: int, present: tuple[int, ...] | None = None) -> Path:
    """An over-cap log L and archives A, B, C, D in slots 1..keep (or only the slots in `present`)."""
    log = _make_log(directory)
    for n, content in zip(range(1, keep + 1), "ABCD", strict=False):
        if present is None or n in present:
            _make_archive(log, n, content)
    return log


def _state(log: Path) -> dict[str, bytes]:
    """The visible state, decompressed: archive slot -> content, plus the active log."""
    out: dict[str, bytes] = {"log": log.read_bytes()}
    for p in sorted(log.parent.iterdir()):
        m = re.fullmatch(re.escape(log.name) + r"\.(\d+)\.gz", p.name)
        if m:
            out[m.group(1)] = _gz_bytes(p)
    return out


def _extras(log: Path) -> list[str]:
    """Anything beyond the log, its lock sidecar and the numbered archives: a temp or a staging file."""
    return sorted(
        p.name
        for p in log.parent.iterdir()
        if p.name != log.name
        and p.name != log.name + ".lock"
        and not re.fullmatch(re.escape(log.name) + r"\.\d+\.gz", p.name)
    )


class Faults:
    """Fail `os.replace` calls whose destination ends with `suffix` (persistently while `on`)."""

    def __init__(self, monkeypatch):
        self.rules: list[tuple[str, BaseException | type, int | None]] = []
        self.calls = 0
        self.kill_at: int | None = None
        real = os.replace

        def replace(src, dst):
            self.calls += 1
            if self.kill_at is not None and self.calls == self.kill_at:
                raise SimulatedKill()
            for i, (suffix, exc, left) in enumerate(self.rules):
                if str(dst).endswith(suffix) and (left is None or left > 0):
                    if left is not None:
                        self.rules[i] = (suffix, exc, left - 1)
                    raise exc if isinstance(exc, BaseException) else exc()
            return real(src, dst)

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr("time.sleep", lambda _s: None)

    def fail(self, suffix: str, exc=None, times: int | None = None):
        self.rules.append((suffix, exc or OSError(errno.EIO, "I/O error"), times))

    def clear(self):
        self.rules.clear()


def _tick(log: Path, keep: int):
    return lr.rotate_rolling_size(log, max_bytes=CAP, archive_keep=keep)


def _clean_final(tmp_path: Path, keep: int, present: tuple[int, ...] | None = None) -> dict[str, bytes]:
    d = tmp_path.with_name(tmp_path.name + "_clean")
    d.mkdir(exist_ok=True)
    log = _setup(d, keep, present)
    _tick(log, keep)
    return _state(log)


def _original(log: Path) -> bytes:
    return log.read_bytes()


# --- the clean path ----------------------------------------------------------------------------------------------


def test_a_clean_rotation_archives_the_log_and_shifts_the_old_archives(tmp_path):
    log = _setup(tmp_path, 3)
    before = log.read_bytes()
    _tick(log, 3)
    state = _state(log)
    assert state["1"] == before and state["2"] == b"A" and state["3"] == b"B"
    assert state["log"] == b"" and _extras(log) == []


def test_a_clean_rotation_retries_transient_permission_errors_on_the_publish(tmp_path, monkeypatch):
    log = _setup(tmp_path, 3)
    before = log.read_bytes()
    faults = Faults(monkeypatch)
    faults.fail(".1.gz", PermissionError(32, "sharing violation"), times=2)
    _tick(log, 3)
    assert _state(log)["1"] == before and _extras(log) == []


# --- (a) the new archive cannot be written -----------------------------------------------------------------------


@pytest.mark.parametrize("how", ["staging-replace", "compress"])
def test_g17a_a_failed_archive_write_three_ticks_in_a_row_loses_nothing(tmp_path, monkeypatch, how):
    log = _setup(tmp_path, 3)
    start = _state(log)
    faults = Faults(monkeypatch)
    if how == "staging-replace":
        faults.fail(".next.gz")
    else:
        real_open = gzip.open

        def boom(*a, **k):
            mode = a[1] if len(a) > 1 else k.get("mode", "rb")
            if "w" in mode:  # only the compression of the new archive fails; reading archives still works
                raise OSError(errno.EIO, "I/O error")
            return real_open(*a, **k)

        monkeypatch.setattr(lr.gzip, "open", boom)
    for _ in range(3):
        with pytest.raises(OSError):
            _tick(log, 3)
        assert _state(log) == start and _extras(log) == []
    monkeypatch.undo()
    _tick(log, 3)
    assert _state(log) == _clean_final(tmp_path, 3)


# --- (b) the publish fails ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [OSError(errno.EIO, "I/O error"), PermissionError(32, "held")])
def test_g17b_a_failed_publish_loses_only_the_one_normal_eviction(tmp_path, monkeypatch, exc):
    log = _setup(tmp_path, 3)
    orig = log.read_bytes()
    faults = Faults(monkeypatch)
    faults.fail(".1.gz", exc)
    for _ in range(3):
        with pytest.raises(type(exc)):
            _tick(log, 3)
        state = _state(log)
        assert state["log"] == orig and state["2"] == b"A" and state["3"] == b"B" and "1" not in state
    faults.clear()
    _tick(log, 3)
    assert _state(log) == _clean_final(tmp_path, 3)


# --- (c) a shift rename fails ------------------------------------------------------------------------------------


def test_g17c_a_failed_first_shift_loses_nothing(tmp_path, monkeypatch):
    log = _setup(tmp_path, 3)
    start = _state(log)
    faults = Faults(monkeypatch)
    faults.fail(".3.gz")
    for _ in range(3):
        with pytest.raises(OSError):
            _tick(log, 3)
        assert _state(log) == start
    faults.clear()
    _tick(log, 3)
    assert _state(log) == _clean_final(tmp_path, 3)


def test_g17c_a_failed_middle_shift_loses_only_the_one_normal_eviction(tmp_path, monkeypatch):
    log = _setup(tmp_path, 4)
    orig = log.read_bytes()
    faults = Faults(monkeypatch)
    faults.fail(".3.gz", times=None)  # the second move of a four-slot shift fails every time
    for tick in range(3):
        with pytest.raises(OSError):
            _tick(log, 4)
        state = _state(log)
        assert state["log"] == orig
        # tick 1 moved D out of the way (the normal eviction), then stopped; nothing else is ever lost
        assert {k: v for k, v in state.items() if k != "log"} == {"1": b"A", "2": b"B", "4": b"C"}, tick
    faults.clear()
    _tick(log, 4)
    assert _state(log) == _clean_final(tmp_path, 4)


# --- (d) a kill at every point ----------------------------------------------------------------------------------


@pytest.mark.parametrize("kill_at", [1, 2, 3, 4])
def test_g17d_a_kill_before_each_replace_then_one_clean_tick_equals_a_clean_run(tmp_path, monkeypatch, kill_at):
    log = _setup(tmp_path, 3)
    orig = log.read_bytes()
    faults = Faults(monkeypatch)
    faults.kill_at = kill_at
    with pytest.raises(SimulatedKill):
        _tick(log, 3)
    assert log.read_bytes() == orig  # the active log is only truncated after the publish
    faults.kill_at = None
    _tick(log, 3)
    assert _state(log) == _clean_final(tmp_path, 3)
    assert _extras(log) == []


def test_g17d_a_kill_inside_the_staging_write_leaves_a_temp_that_the_next_tick_sweeps(tmp_path, monkeypatch):
    log = _setup(tmp_path, 3)
    orig = log.read_bytes()

    def kill(fd):
        raise SimulatedKill()

    monkeypatch.setattr(os, "fsync", kill)
    with pytest.raises(SimulatedKill):
        _tick(log, 3)
    monkeypatch.undo()
    assert log.read_bytes() == orig and {"1", "2", "3"} <= set(_state(log))
    _tick(log, 3)
    assert _state(log) == _clean_final(tmp_path, 3)
    assert _extras(log) == []


def test_g17d_a_kill_between_the_publish_and_the_truncation_is_the_named_residual(tmp_path, monkeypatch):
    """Documented limit accepted by main (2026-10-20 09:20): the rows are archived twice, none is lost, and the
    duplicate displaces the oldest archive one rotation early. The base has the same window."""
    log = _setup(tmp_path, 3)
    orig = log.read_bytes()
    real = Path.write_bytes

    def kill_on_truncate(self, data):
        if self == log and data == b"":
            raise SimulatedKill()
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", kill_on_truncate)
    with pytest.raises(SimulatedKill):
        _tick(log, 3)
    monkeypatch.setattr(Path, "write_bytes", real)
    assert _state(log)["1"] == orig and log.read_bytes() == orig
    _tick(log, 3)
    state = _state(log)
    assert state["1"] == orig and state["2"] == orig and state["3"] == b"A" and state["log"] == b""


# --- (e) archive_keep=1 -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".next.gz", ".1.gz"])
def test_g17e_with_one_slot_a_failed_write_or_publish_keeps_the_single_archive(tmp_path, monkeypatch, suffix):
    log = _setup(tmp_path, 1)
    start = _state(log)
    faults = Faults(monkeypatch)
    faults.fail(suffix)
    for _ in range(3):
        with pytest.raises(OSError):
            _tick(log, 1)
        assert _state(log)["1"] == b"A" and _state(log)["log"] == start["log"]
    faults.clear()
    _tick(log, 1)
    assert _state(log) == _clean_final(tmp_path, 1)


# --- (f) ENOSPC -------------------------------------------------------------------------------------------------


def _enospc():
    return OSError(errno.ENOSPC, "No space left on device")


def test_g17f_persistent_enospc_evicts_only_the_one_archive_a_rotation_would_evict(tmp_path, monkeypatch):
    log = _setup(tmp_path, 3)
    orig = log.read_bytes()
    faults = Faults(monkeypatch)
    faults.fail(".next.gz", _enospc())
    for tick in range(3):
        with pytest.raises(OSError):
            _tick(log, 3)
        state = _state(log)
        assert state["log"] == orig
        assert {k: v for k, v in state.items() if k != "log"} == {"1": b"A", "2": b"B"}, tick
    faults.clear()
    _tick(log, 3)
    state = _state(log)
    assert state["1"] == orig and state["2"] == b"A" and state["3"] == b"B"


def test_g17f_a_single_enospc_is_retried_once_and_equals_a_clean_rotation(tmp_path, monkeypatch):
    log = _setup(tmp_path, 3)
    faults = Faults(monkeypatch)
    faults.fail(".next.gz", _enospc(), times=1)
    _tick(log, 3)
    assert _state(log) == _clean_final(tmp_path, 3)


def test_g17f_enospc_while_a_gap_exists_evicts_nothing(tmp_path, monkeypatch):
    log = _setup(tmp_path, 3, present=(1, 2))
    faults = Faults(monkeypatch)
    faults.fail(".next.gz", _enospc())
    with pytest.raises(OSError):
        _tick(log, 3)
    state = _state(log)
    assert {k: v for k, v in state.items() if k != "log"} == {"1": b"A", "2": b"B"}


def test_g17f_enospc_with_one_slot_evicts_the_one_archive_once(tmp_path, monkeypatch):
    log = _setup(tmp_path, 1)
    faults = Faults(monkeypatch)
    faults.fail(".next.gz", _enospc())
    for _ in range(2):
        with pytest.raises(OSError):
            _tick(log, 1)
        assert "1" not in _state(log)


# --- (g) a stale staging file -----------------------------------------------------------------------------------


def test_g17g_a_stale_staging_file_is_overwritten_and_never_an_archive(tmp_path):
    log = _setup(tmp_path, 3)
    orig = log.read_bytes()
    stale = log.parent / (log.name + ".next.gz")
    stale.write_bytes(b"not a gzip file: left by a killed rotation")
    assert not re.fullmatch(re.escape(log.name) + r"\.\d+\.gz", stale.name)
    _tick(log, 3)
    state = _state(log)
    assert state["1"] == orig and state["2"] == b"A" and state["3"] == b"B"
    assert _extras(log) == []  # the staging file was replaced onto .1.gz, none is left


# --- the year archive (S11) and the compress seams --------------------------------------------------------------


def test_the_year_archive_is_written_with_lf_and_keeps_every_row(tmp_path):
    import json
    from datetime import UTC, datetime

    log = tmp_path / "soul_audit.jsonl"
    rows = [{"ts": "2025-12-01T00:00:00+00:00", "i": 1}, {"ts": "2026-02-01T00:00:00+00:00", "i": 2}]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lr.rotate_age_archive_yearly(log, now=datetime(2026, 6, 14, tzinfo=UTC), timestamp_field="ts")
    raw = _gz_bytes(tmp_path / "soul_audit.2025.jsonl.gz")
    assert b"\r" not in raw and json.loads(raw.decode().splitlines()[0])["i"] == 1
    assert [e for e in os.listdir(tmp_path) if e.endswith((".tmp", ".new"))] == []


def test_the_rolling_compress_still_goes_through_gzip_open_and_shutil_copyfileobj():
    """The two seams that existing tests pin (test_log_rotation.py, test_log_rotation_locking_476.py)."""
    import ast

    src = Path(lr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_rotate_rolling_locked")
    called = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "gzip.open" in called and "shutil.copyfileobj" in called
