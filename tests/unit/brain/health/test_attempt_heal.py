"""Tests for brain.health.attempt_heal — core heal + save helpers."""

from __future__ import annotations

import ast
import errno
import json
import os
import shutil
import stat
import sys
import threading
from pathlib import Path

import pytest

import brain.health.attempt_heal as heal
from brain.health.attempt_heal import (
    attempt_heal,
    save_with_backup,
    save_with_backup_text,
)


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


# ---- save_with_backup_text — raw-text variant ----


def test_save_with_backup_text_first_save_no_bak(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    save_with_backup_text(p, "# title\n\nbody")
    assert p.read_text(encoding="utf-8") == "# title\n\nbody"
    assert not (tmp_path / "x.md.bak1").exists()
    assert not (tmp_path / "x.md.new").exists()


def test_save_with_backup_text_rotates_3_levels(tmp_path: Path) -> None:
    """Rotation pattern matches save_with_backup but with raw text."""
    p = tmp_path / "x.md"
    save_with_backup_text(p, "v1")
    save_with_backup_text(p, "v2")
    save_with_backup_text(p, "v3")
    save_with_backup_text(p, "v4")
    assert p.read_text(encoding="utf-8") == "v4"
    assert (tmp_path / "x.md.bak1").read_text(encoding="utf-8") == "v3"
    assert (tmp_path / "x.md.bak2").read_text(encoding="utf-8") == "v2"
    assert (tmp_path / "x.md.bak3").read_text(encoding="utf-8") == "v1"


def test_save_with_backup_text_caps_at_3_drops_oldest(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    for i in range(1, 6):
        save_with_backup_text(p, f"v{i}")
    assert p.read_text(encoding="utf-8") == "v5"
    assert (tmp_path / "x.md.bak3").read_text(encoding="utf-8") == "v2"
    assert not (tmp_path / "x.md.bak4").exists()


def test_save_with_backup_text_unlinks_stale_new(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    (tmp_path / "x.md.new").write_text("stale partial", encoding="utf-8")
    save_with_backup_text(p, "fresh")
    assert not (tmp_path / "x.md.new").exists()
    assert p.read_text(encoding="utf-8") == "fresh"


def test_save_with_backup_text_writes_raw_not_json(tmp_path: Path) -> None:
    """The text variant must not JSON-encode. A string containing quotes,
    backslashes, and newlines must round-trip byte-identical — proving this
    isn't save_with_backup with the JSON path stripped."""
    p = tmp_path / "x.md"
    raw = 'hana said "i love you" — three\\n words\nwith newline'
    save_with_backup_text(p, raw)
    assert p.read_text(encoding="utf-8") == raw


def test_save_with_backup_delegates_to_text_for_json_payload(tmp_path: Path) -> None:
    """save_with_backup is now a thin wrapper around save_with_backup_text.
    The wrapper handles JSON encoding + trailing newline; the text helper
    does the rotation. This test pins the contract: identical files when
    you pre-encode."""
    p_dict = tmp_path / "via_dict.json"
    p_text = tmp_path / "via_text.json"
    save_with_backup(p_dict, {"a": 1, "b": [2, 3]})
    save_with_backup_text(p_text, json.dumps({"a": 1, "b": [2, 3]}, indent=2) + "\n")
    assert p_dict.read_text(encoding="utf-8") == p_text.read_text(encoding="utf-8")


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


# ---- #200: Windows WinError 32 — rename while another handle reads the file ----


def test_save_with_backup_text_retries_transient_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Windows refuses os.replace on a file another handle has open (WinError 32).

    The supervisor reads persona_config.json inside its ticks in the same
    process, so a settings write can collide with a read. A transient
    PermissionError on the rotate/replace must be retried, not surfaced.
    """
    import os

    import brain.health.attempt_heal as heal

    p = tmp_path / "x.json"
    save_with_backup_text(p, "v1")
    real_replace = os.replace
    failures = {"left": 2}

    def flaky_replace(src, dst):
        if failures["left"] > 0:
            failures["left"] -= 1
            raise PermissionError(32, "The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(heal.os, "replace", flaky_replace)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)

    save_with_backup_text(p, "v2")
    assert p.read_text(encoding="utf-8") == "v2"
    assert (tmp_path / "x.json.bak1").read_text(encoding="utf-8") == "v1"
    assert failures["left"] == 0


def test_save_with_backup_text_reraises_persistent_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A real permission problem must still fail loud after the retry budget."""
    import brain.health.attempt_heal as heal

    p = tmp_path / "x.json"
    save_with_backup_text(p, "v1")

    def always_denied(src, dst):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(heal.os, "replace", always_denied)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        save_with_backup_text(p, "v2")


# ---- #387: the live path must never be absent while a save rotates the backups ----
#
# One recorder fires a hook before and after every filesystem mutation save_with_backup_text makes
# (os.replace, os.fsync, Path.unlink, a writable os.open), so the property is checked at every step
# boundary deterministically instead of hoping a thread lands in a microsecond window.


class SimulatedCrash(BaseException):
    """Stands in for a kill or power loss. Not an Exception: cleanup that a real kill would skip must
    not run for it, so the temp files a real crash leaves behind are left behind here too."""


def _payload(n: int) -> str:
    return json.dumps({"v": n, "pad": "x" * 2000}) + "\n"


def _version(path: Path) -> int:
    return json.loads(path.read_text(encoding="utf-8"))["v"]


def _seed(path: Path, upto: int = 5, count: int = 3) -> None:
    for n in range(1, upto + 1):
        save_with_backup_text(path, _payload(n), backup_count=count)


def _fd_name(fd: int, root: Path) -> str:
    ino = os.fstat(fd).st_ino
    if root.stat().st_ino == ino:
        return "<dir>"
    for f in root.iterdir():
        if f.stat().st_ino == ino:
            return f.name
    return "<unknown>"


class _Recorder:
    """hook(phase, op, name) runs before and after each mutation under root; it may raise."""

    def __init__(self, monkeypatch, root: Path, hook=None) -> None:
        self.mp = monkeypatch
        self.root = root
        self.hook = hook
        self.events: list[tuple[str, str, str]] = []

    def _fire(self, phase: str, op: str, name: str) -> None:
        self.events.append((phase, op, name))
        if self.hook is not None:
            self.hook(phase, op, name)

    def _inside(self, p) -> bool:
        try:
            return Path(os.fspath(p)).parent == self.root
        except TypeError:
            return False

    def install(self) -> None:
        real_replace, real_fsync, real_open = os.replace, os.fsync, os.open
        real_unlink = Path.unlink
        rec = self

        def replace(src, dst):
            if not rec._inside(dst):
                return real_replace(src, dst)
            name = f"{Path(src).name}->{Path(dst).name}"
            rec._fire("before", "replace", name)
            real_replace(src, dst)
            rec._fire("after", "replace", name)

        def fsync(fd):
            name = _fd_name(fd, rec.root)
            if name == "<unknown>":
                return real_fsync(fd)
            rec._fire("before", "fsync", name)
            real_fsync(fd)
            rec._fire("after", "fsync", name)

        def unlink(self, missing_ok=False):
            if not rec._inside(self):
                return real_unlink(self, missing_ok=missing_ok)
            rec._fire("before", "unlink", self.name)
            real_unlink(self, missing_ok=missing_ok)
            rec._fire("after", "unlink", self.name)

        def os_open(path, flags, *args, **kwargs):
            writable = flags & (os.O_WRONLY | os.O_RDWR)
            if not writable or not rec._inside(path):
                return real_open(path, flags, *args, **kwargs)
            name = Path(path).name
            rec._fire("before", "open", name)
            fd = real_open(path, flags, *args, **kwargs)
            try:
                rec._fire("after", "open", name)
            except BaseException:
                os.close(fd)
                raise
            return fd

        self.mp.setattr(os, "replace", replace)
        self.mp.setattr(os, "fsync", fsync)
        self.mp.setattr(os, "open", os_open)
        self.mp.setattr(Path, "unlink", unlink)


def _record_one_save(monkeypatch, tmp_path: Path, name: str) -> list[tuple[str, str, str]]:
    """The event list of one v6 save over a seeded full chain, in a scratch dir."""
    d = tmp_path / name
    d.mkdir()
    p = d / "x.json"
    _seed(p)
    rec = _Recorder(monkeypatch, d)
    rec.install()
    save_with_backup_text(p, _payload(6))
    monkeypatch.undo()
    return rec.events


def test_live_path_never_missing_at_any_step_boundary(tmp_path: Path, monkeypatch) -> None:
    """C1: at every boundary of a save the live path holds a complete payload (old until the final
    replace has run, new after)."""
    d = tmp_path / "c1"
    d.mkdir()
    p = d / "x.json"
    _seed(p)
    state = {"swapped": False}
    problems: list[str] = []

    def hook(phase: str, op: str, name: str) -> None:
        if phase == "after" and op == "replace" and name.endswith("->x.json"):
            state["swapped"] = True
        want = 6 if state["swapped"] else 5
        if not p.exists():
            problems.append(f"live missing at {phase} {op} {name}")
            return
        try:
            got = _version(p)
        except (ValueError, KeyError) as exc:
            problems.append(f"live unreadable at {phase} {op} {name}: {exc}")
            return
        if got != want:
            problems.append(f"live v{got}, wanted v{want} at {phase} {op} {name}")

    _Recorder(monkeypatch, d, hook).install()
    save_with_backup_text(p, _payload(6))
    monkeypatch.undo()
    assert not problems, problems
    assert state["swapped"]
    assert _version(p) == 6 and _version(d / "x.json.bak1") == 5


@pytest.mark.skipif(
    os.name == "nt",
    reason="a reader holding the file open without FILE_SHARE_DELETE makes any replace fail on "
    "Windows by design (#200); there the deterministic boundary and crash tests carry the property",
)
@pytest.mark.parametrize("run", [1, 2, 3])
def test_concurrent_reader_never_sees_live_path_missing(tmp_path: Path, monkeypatch, run) -> None:
    """C2: a thread polling the live path through 1500 saves sees no gap, no partial file, and
    versions that never go backwards."""
    p = tmp_path / "x.json"
    save_with_backup_text(p, _payload(0))
    monkeypatch.setattr(os, "fsync", lambda fd: None)  # durability does not change the race
    stop = threading.Event()
    stats = {"reads": 0, "missing": 0, "bad": 0, "backwards": 0}

    def poll() -> None:
        last = -1
        while not stop.is_set():
            try:
                v = json.loads(p.read_text(encoding="utf-8"))["v"]
            except FileNotFoundError:
                stats["missing"] += 1
                continue
            except (ValueError, KeyError, OSError):
                stats["bad"] += 1
                continue
            stats["reads"] += 1
            if v < last:
                stats["backwards"] += 1
            last = v

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    t = threading.Thread(target=poll)
    t.start()
    try:
        for n in range(1, 1501):
            save_with_backup_text(p, _payload(n))
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old_interval)
    assert stats["reads"] > 100, stats  # the poller really ran
    assert stats["missing"] == 0 and stats["bad"] == 0 and stats["backwards"] == 0, stats


def test_crash_at_every_step_leaves_a_complete_live_file(tmp_path: Path, monkeypatch) -> None:
    """C3: kill at every before/after boundary of every mutation, and mid-write of the temp file.
    Live is always one complete version, every backup at a final name is complete, the heal sees
    no anomaly, and the next save cleans up stale temps and rotates correctly."""
    total = len(_record_one_save(monkeypatch, tmp_path, "dry"))
    assert total > 0
    problems: list[str] = []
    left_bak1_new = 0

    def check(d: Path, label: str) -> None:
        nonlocal left_bak1_new
        p = d / "x.json"
        if not p.exists():
            problems.append(f"{label}: live missing")
            return
        try:
            live = _version(p)
        except (ValueError, KeyError) as exc:
            problems.append(f"{label}: live unreadable ({exc})")
            return
        if live not in (5, 6):
            problems.append(f"{label}: live is v{live}, neither old nor new")
        for i in (1, 2, 3):
            bak = d / f"x.json.bak{i}"
            if bak.exists():
                try:
                    _version(bak)
                except (ValueError, KeyError) as exc:
                    problems.append(f"{label}: bak{i} at final name is incomplete ({exc})")
        data, anomaly = attempt_heal(p, _default)
        if anomaly is not None or data["v"] != live:
            problems.append(f"{label}: heal saw anomaly={anomaly} data={data}")
        if (d / "x.json.bak1.new").exists():
            left_bak1_new += 1
        save_with_backup_text(p, _payload(7))
        if (d / "x.json.new").exists() or (d / "x.json.bak1.new").exists():
            problems.append(f"{label}: next save left a stale temp")
        if _version(p) != 7 or _version(d / "x.json.bak1") != live:
            problems.append(f"{label}: next save chain wrong")

    for j in range(total):
        d = tmp_path / f"crash{j}"
        d.mkdir()
        _seed(d / "x.json")
        counter = {"n": 0}

        def hook(phase: str, op: str, name: str, j=j, counter=counter) -> None:
            if counter["n"] == j:
                counter["n"] += 1
                raise SimulatedCrash(f"{phase} {op} {name}")
            counter["n"] += 1

        _Recorder(monkeypatch, d, hook).install()
        with pytest.raises(SimulatedCrash) as info:
            save_with_backup_text(d / "x.json", _payload(6))
        monkeypatch.undo()
        if str(info.value) == "after open x.json.bak1.new":
            # A real kill runs no cleanup code, so the empty temp must still be there.
            if not (d / "x.json.bak1.new").exists():
                problems.append(f"crash at #{j}: cleanup ran for a simulated kill")
        check(d, f"crash at #{j} ({info.value})")

    # A write of the temp file that stops half way.
    d = tmp_path / "half"
    d.mkdir()
    _seed(d / "x.json")
    real_open = Path.open

    class _Half:
        def __init__(self, f) -> None:
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self.f.close()

        def write(self, data: str) -> int:
            self.f.write(data[: len(data) // 2])
            self.f.flush()
            raise SimulatedCrash("half write")

        def __getattr__(self, name: str):
            return getattr(self.f, name)

    def half_open(self, *args, **kwargs):
        f = real_open(self, *args, **kwargs)
        return _Half(f) if self.name.endswith(".new") else f

    monkeypatch.setattr(Path, "open", half_open)
    with pytest.raises(SimulatedCrash):
        save_with_backup_text(d / "x.json", _payload(6))
    monkeypatch.undo()
    check(d, "half-written temp")

    assert not problems, problems
    assert left_bak1_new > 0, "no crash point left a stale .bak1.new, so its cleanup went untested"


@pytest.mark.parametrize("count", [0, 1, 2, 3, 6])
def test_backup_chain_matches_model_for_each_count(tmp_path: Path, count: int) -> None:
    """C4: .bak{i} holds version n-i for i in 1..max(count, 1), nothing older, no temp files."""
    p = tmp_path / "x.json"
    n_saves = 10
    for n in range(1, n_saves + 1):
        save_with_backup_text(p, _payload(n), backup_count=count)
    kept = max(count, 1)
    assert _version(p) == n_saves
    for i in range(1, kept + 1):
        assert _version(tmp_path / f"x.json.bak{i}") == n_saves - i
    assert not (tmp_path / f"x.json.bak{kept + 1}").exists()
    assert not (tmp_path / "x.json.new").exists()
    assert not (tmp_path / "x.json.bak1.new").exists()


def test_transient_permission_error_at_every_replace_is_retried(
    tmp_path: Path, monkeypatch
) -> None:
    """C5a (replaces): a transient WinError-32 style failure at any one replace is absorbed."""
    total = sum(
        1 for e in _record_one_save(monkeypatch, tmp_path, "dry") if e[:2] == ("before", "replace")
    )
    assert total >= 3
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    real_replace = os.replace
    for n in range(1, total + 1):
        d = tmp_path / f"r{n}"
        d.mkdir()
        p = d / "x.json"
        _seed(p)
        calls = {"n": 0}

        def flaky(src, dst, n=n, calls=calls, d=d):
            if Path(dst).parent == d:
                calls["n"] += 1
                if calls["n"] == n:
                    raise PermissionError(32, "The process cannot access the file")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky)
        save_with_backup_text(p, _payload(6))
        monkeypatch.undo()
        monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
        assert _version(p) == 6 and _version(d / "x.json.bak1") == 5, f"replace #{n}"


@pytest.mark.parametrize("point", ["before_create", "after_create", "at_copystat"])
def test_transient_permission_error_inside_backup_copy_is_retried(
    tmp_path: Path, monkeypatch, point: str
) -> None:
    """C5a (copy): the backup copy retries too, and an attempt that follows a half-made temp
    does not trip over it (no FileExistsError)."""
    p = tmp_path / "x.json"
    _seed(p)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    fired = {"n": 0}
    real_open, real_copystat = os.open, shutil.copystat

    def os_open(path, flags, *args, **kwargs):
        if Path(path).name != "x.json.bak1.new" or fired["n"]:
            return real_open(path, flags, *args, **kwargs)
        if point == "before_create":
            fired["n"] += 1
            raise PermissionError(32, "busy")
        if point == "after_create":
            fd = real_open(path, flags, *args, **kwargs)
            os.close(fd)
            fired["n"] += 1
            raise PermissionError(32, "busy")
        return real_open(path, flags, *args, **kwargs)

    def copystat(src, dst, **kwargs):
        if point == "at_copystat" and not fired["n"] and Path(dst).name == "x.json.bak1.new":
            fired["n"] += 1
            raise PermissionError(32, "busy")
        return real_copystat(src, dst, **kwargs)

    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(shutil, "copystat", copystat)
    save_with_backup_text(p, _payload(6))
    monkeypatch.undo()
    assert fired["n"] == 1, "the injection point was never reached"
    assert _version(p) == 6 and _version(tmp_path / "x.json.bak1") == 5
    assert not (tmp_path / "x.json.bak1.new").exists()


def test_persistent_failure_of_final_replace_leaves_old_live_and_intact_chain(
    tmp_path: Path, monkeypatch
) -> None:
    """C5b: only new->live keeps failing. The save raises, live is still the complete old version,
    bak1 holds the old live content and bak2 the previous bak1 (never a missing live file)."""
    p = tmp_path / "x.json"
    _seed(p)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    real_replace = os.replace

    def deny_final(src, dst):
        if Path(dst) == p:
            raise PermissionError(32, "The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", deny_final)
    with pytest.raises(PermissionError):
        save_with_backup_text(p, _payload(6))
    monkeypatch.undo()
    assert p.exists(), "live file went missing"
    assert _version(p) == 5
    assert _version(tmp_path / "x.json.bak1") == 5
    assert _version(tmp_path / "x.json.bak2") == 4


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(heal.__file__).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_save_helper_has_no_bare_replace_or_side_door_rename() -> None:
    """C5d / C13: every replace in save_with_backup_text goes through _replace_with_retry, and the
    live file is never moved by another rename API."""
    forbidden = {"os.replace", "os.rename", "shutil.move", "os.renames"}
    attr_forbidden = {"rename", "replace"}
    for node in ast.walk(_function("save_with_backup_text")):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        dotted = ast.unparse(fn)
        assert dotted not in forbidden, dotted
        if isinstance(fn, ast.Attribute):
            assert fn.attr not in attr_forbidden, dotted


def test_fsync_of_new_file_and_backup_copy_precede_their_replaces(
    tmp_path: Path, monkeypatch
) -> None:
    """C6 (proxy for power-loss durability, which cannot be simulated here): the temp file is fully
    on disk and fsynced before the replace that publishes it, for both the new live file and the
    backup copy."""
    p = tmp_path / "x.json"
    _seed(p)
    old_bytes = p.read_bytes()
    text = _payload(6)
    seen: dict[str, bytes] = {}

    def hook(phase: str, op: str, name: str) -> None:
        if phase == "before" and op == "fsync" and name.endswith(".new"):
            seen[name] = (tmp_path / name).read_bytes()

    rec = _Recorder(monkeypatch, tmp_path, hook)
    rec.install()
    save_with_backup_text(p, text)
    monkeypatch.undo()
    order = [(op, name) for phase, op, name in rec.events if phase == "before"]
    assert ("fsync", "x.json.new") in order, order
    assert order.index(("fsync", "x.json.new")) < order.index(("replace", "x.json.new->x.json"))
    assert seen["x.json.new"] == text.encode("utf-8")
    assert ("fsync", "x.json.bak1.new") in order, order
    assert order.index(("fsync", "x.json.bak1.new")) < order.index(
        ("replace", "x.json.bak1.new->x.json.bak1")
    )
    assert seen["x.json.bak1.new"] == old_bytes


def test_backup_is_byte_exact_and_keeps_mode_and_mtime(tmp_path: Path, monkeypatch) -> None:
    """C7: the backup carries what the rename used to carry: bytes, mtime, and (POSIX) mode, and the
    half-made copy is never wider than its source."""
    p = tmp_path / "x.json"
    save_with_backup_text(p, "line1\r\nline2\n")
    os.utime(p, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    if os.name != "nt":
        p.chmod(0o640)
    before = p.read_bytes()
    before_stat = p.stat()
    modes: list[int] = []

    def hook(phase: str, op: str, name: str) -> None:
        tmp = tmp_path / "x.json.bak1.new"
        if phase == "after" and tmp.exists():
            modes.append(stat.S_IMODE(tmp.stat().st_mode))

    _Recorder(monkeypatch, tmp_path, hook).install()
    save_with_backup_text(p, "next\n")
    monkeypatch.undo()
    bak = tmp_path / "x.json.bak1"
    assert bak.read_bytes() == before
    assert bak.stat().st_mtime_ns == before_stat.st_mtime_ns
    if os.name != "nt":
        assert stat.S_IMODE(bak.stat().st_mode) == 0o640
        assert modes, "the backup temp file was never observed"
        assert all(m & ~0o640 == 0 for m in modes), [oct(m) for m in modes]


@pytest.mark.parametrize("point", ["after_open", "at_fsync"])
def test_failed_backup_copy_leaves_chain_untouched(tmp_path: Path, monkeypatch, point: str) -> None:
    """C8: disk full while making the backup copy. Nothing was rotated yet: live and every backup
    are byte-identical, the half-made copy is removed, and the next save works."""
    p = tmp_path / "x.json"
    _seed(p)
    names = ["x.json", "x.json.bak1", "x.json.bak2", "x.json.bak3"]
    before = {n: (tmp_path / n).read_bytes() for n in names}

    def hook(phase: str, op: str, name: str) -> None:
        if name != "x.json.bak1.new":
            return
        if (point == "after_open" and (phase, op) == ("after", "open")) or (
            point == "at_fsync" and (phase, op) == ("before", "fsync")
        ):
            raise OSError(errno.ENOSPC, "No space left on device")

    _Recorder(monkeypatch, tmp_path, hook).install()
    with pytest.raises(OSError) as info:
        save_with_backup_text(p, _payload(6))
    monkeypatch.undo()
    assert info.value.errno == errno.ENOSPC
    assert {n: (tmp_path / n).read_bytes() for n in names} == before
    assert not (tmp_path / "x.json.bak1.new").exists()
    save_with_backup_text(p, _payload(6))
    assert not (tmp_path / "x.json.new").exists()
    assert _version(p) == 6 and _version(tmp_path / "x.json.bak1") == 5


def test_stale_temp_files_are_removed_and_never_used(tmp_path: Path) -> None:
    """C9: leftovers of a killed save are cleaned up and never become a backup or the live file."""
    p = tmp_path / "x.json"
    _seed(p)
    (tmp_path / "x.json.new").write_text("garbage-new", encoding="utf-8")
    (tmp_path / "x.json.bak1.new").write_text("garbage-bak", encoding="utf-8")
    save_with_backup_text(p, _payload(6))
    assert _version(p) == 6 and _version(tmp_path / "x.json.bak1") == 5
    assert not (tmp_path / "x.json.new").exists()
    assert not (tmp_path / "x.json.bak1.new").exists()


def test_stale_backup_temp_is_removed_even_on_a_first_save(tmp_path: Path) -> None:
    """C9: with no live file there is no backup to make, but a stale backup temp still goes."""
    p = tmp_path / "x.json"
    (tmp_path / "x.json.bak1.new").write_text("garbage-bak", encoding="utf-8")
    save_with_backup_text(p, _payload(1))
    assert not (tmp_path / "x.json.bak1.new").exists()
    assert not (tmp_path / "x.json.bak1").exists()


def test_new_file_is_written_in_text_mode_exactly_like_write_text(
    tmp_path: Path, monkeypatch
) -> None:
    """C10: same open mode and encoding, and no newline argument, so Windows keeps its CRLF
    translation; the bytes equal what Path.write_text produces."""
    p = tmp_path / "x.json"
    calls: list[tuple[tuple, dict]] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        if self.name == "x.json.new":
            calls.append((args, kwargs))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    text = "alpha\nbeta \u00e9\n"
    save_with_backup_text(p, text)
    monkeypatch.undo()
    assert len(calls) == 1
    args, kwargs = calls[0]
    mode = args[0] if args else kwargs.get("mode")
    assert mode == "w"
    assert kwargs.get("encoding") == "utf-8"
    assert kwargs.get("newline") is None
    ref = tmp_path / "ref.txt"
    ref.write_text(text, encoding="utf-8")
    assert p.read_bytes() == ref.read_bytes()


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permission bits and a non-root user",
)
def test_unreadable_live_file_fails_loud_and_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    """C14: a live file the saver cannot read makes the save raise (a real permission problem fails
    loud after the retry budget) without touching live, the backups, or leaving a backup temp."""
    p = tmp_path / "x.json"
    _seed(p)
    names = ["x.json", "x.json.bak1", "x.json.bak2", "x.json.bak3"]
    before = {n: (tmp_path / n).read_bytes() for n in names}
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    p.chmod(0)
    try:
        with pytest.raises(PermissionError):
            save_with_backup_text(p, _payload(6))
    finally:
        p.chmod(0o644)
    assert {n: (tmp_path / n).read_bytes() for n in names} == before
    assert not (tmp_path / "x.json.bak1.new").exists()
