"""#472 (C34): lock and marker files judge their holder by process identity, not a bare pid, and older readers still
parse every new-format file."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from brain import process_identity as pi

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux token reader")


@pytest.fixture
def sleeper():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


# ---- the pre-#472 parsers (copied verbatim in spirit from 009cd3b1), standing in for an older version's reader ----


def _old_migrate_holder(text: str) -> int | None:
    try:
        return int(text.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def _old_overlay_lock_holder(text: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return 0


def _old_in_use_pids(folder: Path) -> list[int]:
    return [int(m.name) for m in (folder / ".in-use").iterdir() if m.name.isdigit()]


def _old_compaction_holder(text: str) -> int:
    return int(json.loads(text).get("pid", -1))


_OLD_TEMP = re.compile(r"^(?P<name>.+)\.t(\d+)-[0-9a-f]+\.(?:bak1\.)?new$")


# ---- migrate lock ----


@linux_only
def test_migrate_lock_reused_pid_is_stale_and_live_holder_is_held(tmp_path, sleeper):
    from brain.migrator import cli as mcli

    lock = tmp_path / ".target.migrate.lock"
    lock.write_text(f"{sleeper.pid}\n2026-10-19T00:00:00+00:00\nlinux:other:{sleeper.pid}:1\n", encoding="utf-8")
    fd = mcli._acquire_migrate_lock(lock)  # (a) reused pid: taken over
    os.close(fd)
    assert lock.read_text(encoding="utf-8").splitlines()[0] == str(os.getpid())
    lock.write_text(f"{sleeper.pid}\nts\n{pi.token_of(sleeper.pid)}\n", encoding="utf-8")
    with pytest.raises(RuntimeError):  # (b) the live holder itself: held
        mcli._acquire_migrate_lock(lock)
    lock.write_text(f"{sleeper.pid}\nts\n", encoding="utf-8")
    with pytest.raises(RuntimeError):  # (c) old format: pid-only, live pid -> held
        mcli._acquire_migrate_lock(lock)


def test_migrate_lock_new_format_parses_for_an_old_reader(tmp_path):
    from brain.migrator import cli as mcli

    lock = tmp_path / ".t.migrate.lock"
    fd = mcli._acquire_migrate_lock(lock)
    os.close(fd)
    assert _old_migrate_holder(lock.read_text(encoding="utf-8")) == os.getpid()


# ---- compaction lock ----


@linux_only
def test_compaction_lock_reused_pid_is_reaped_live_holder_kept(tmp_path, sleeper):
    from brain.ingest import buffer

    pd = tmp_path / "p"
    (pd / "active_conversations").mkdir(parents=True)
    path = buffer._compacting_lock_path(pd, "s1")
    path.write_text(json.dumps({"pid": sleeper.pid, "ts": "x", "token": f"linux:other:{sleeper.pid}:1"}))
    assert buffer.acquire_compaction_lock(pd, "s1") is True  # (a)
    path.write_text(json.dumps({"pid": sleeper.pid, "ts": "x", "token": pi.token_of(sleeper.pid)}))
    assert buffer.acquire_compaction_lock(pd, "s1") is False  # (b)
    path.write_text(json.dumps({"pid": sleeper.pid, "ts": "x"}))
    assert buffer.acquire_compaction_lock(pd, "s1") is False  # (c) old format, live pid, fresh mtime


def test_compaction_lock_new_format_parses_for_an_old_reader(tmp_path):
    from brain.ingest import buffer

    pd = tmp_path / "p"
    (pd / "active_conversations").mkdir(parents=True)
    assert buffer.acquire_compaction_lock(pd, "s1") is True
    assert _old_compaction_holder(buffer._compacting_lock_path(pd, "s1").read_text()) == os.getpid()


# ---- overlay in-use marker and update lock ----


@linux_only
def test_in_use_marker_reused_pid_is_not_in_use(tmp_path, sleeper):
    from brain.update import overlay

    folder = tmp_path / "f"
    (folder / ".in-use").mkdir(parents=True)
    (folder / ".in-use" / str(sleeper.pid)).write_text(f"linux:other:{sleeper.pid}:1", encoding="utf-8")
    assert overlay.in_use(folder) is False  # (a)
    (folder / ".in-use" / str(sleeper.pid)).write_text(pi.token_of(sleeper.pid), encoding="utf-8")
    assert overlay.in_use(folder) is True  # (b)
    (folder / ".in-use" / str(sleeper.pid)).write_text("", encoding="utf-8")
    assert overlay.in_use(folder) is True  # (c) old (empty) marker: pid-only
    assert _old_in_use_pids(folder) == [sleeper.pid]  # (e)


@linux_only
def test_overlay_lock_sidecar_rules(tmp_path, sleeper):
    from brain.update import overlay

    root = tmp_path / "root"
    root.mkdir()
    lock = root / overlay.LOCK_FILE
    side = root / (overlay.LOCK_FILE + ".token")
    # (a) live pid, sidecar names it with another token (reused pid): stale, taken over, sidecar replaced
    lock.write_text(str(sleeper.pid), encoding="utf-8")
    side.write_text(f"{sleeper.pid} linux:other:{sleeper.pid}:1", encoding="utf-8")
    with overlay.overlay_lock(root):
        assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())
        assert _old_overlay_lock_holder(lock.read_text(encoding="utf-8")) == os.getpid()  # (e)
        assert side.read_text(encoding="utf-8").split()[0] == str(os.getpid())
    assert not lock.exists() and not side.exists()  # (d) release removes its own lock and sidecar
    # a sidecar whose pid differs from the lock's pid is ignored (pid-only): live lock pid -> held
    lock.write_text(str(sleeper.pid), encoding="utf-8")
    side.write_text(f"{os.getpid() + 1} linux:other:x:1", encoding="utf-8")
    with pytest.raises(overlay.OverlayBusy):
        with overlay.overlay_lock(root):
            pass
    # (b) the live holder's own sidecar: held
    side.write_text(f"{sleeper.pid} {pi.token_of(sleeper.pid)}", encoding="utf-8")
    with pytest.raises(overlay.OverlayBusy):
        with overlay.overlay_lock(root):
            pass


# ---- save temps (attempt_heal, #439's site) ----


@linux_only
def test_save_temp_of_a_reused_pid_is_removed_live_owner_kept(tmp_path, sleeper):
    from brain.health import attempt_heal as ah

    target = tmp_path / "x.json"
    target.write_text("{}", encoding="utf-8")
    digest_other = hashlib.sha1(b"linux:other:1:1").hexdigest()[:12]
    digest_live = hashlib.sha1(pi.token_of(sleeper.pid).encode()).hexdigest()[:12]
    stale = tmp_path / f"x.json.t{sleeper.pid}-k{digest_other}-aaaaaaaaaaaa.new"
    live = tmp_path / f"x.json.t{sleeper.pid}-k{digest_live}-bbbbbbbbbbbb.new"
    old = tmp_path / f"x.json.t{sleeper.pid}-cccccccccccc.new"
    for f in (stale, live, old):
        f.write_text("{}", encoding="utf-8")
    ah._remove_stale_temps(target)
    assert not stale.exists()  # (a)
    assert live.exists()  # (b)
    assert old.exists()  # (c) old shape: pid-only, live pid -> left


def test_new_temp_names_are_skipped_by_an_old_reader():
    name = "x.json.t123-kabcdefabcdef-0123456789ab.new"
    # The old regex needs hex right after "t<pid>-": "-k<digest>-..." breaks it, so an older version leaves the temp.
    assert _OLD_TEMP.match(name) is None
