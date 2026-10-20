"""brain.utils.atomic_write: the one home of the shared atomic writers (#465 C2).

Covers the bytes writer contract, the directory fsync, the content-addressed blob writer, the text writer's
interrupt cleanup, the sweep/in-flight interleaving and the placement rules (one module, no shim, no cycle).
"""

from __future__ import annotations

import ast
import os
import re
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import brain.utils.atomic_write as aw
from brain.utils.atomic_write import (
    fsync_dir,
    save_blob_once,
    save_bytes_atomic,
    save_with_backup_text,
)

REPO = Path(__file__).resolve().parents[4]
TEMP_RE = re.compile(r"\.t\d+-(?:k[0-9a-f]{12}-)?[0-9a-f]+\.(?:bak1\.)?new$")


class SimulatedKill(BaseException):
    """Stands in for a kill: not an Exception, so no cleanup may run for it (as SimulatedCrash in test_attempt_heal)."""


def _temps(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".new"))


def _flaky_replace(monkeypatch, suffix: str, failures: int) -> dict:
    real = os.replace
    state = {"n": failures, "calls": 0}

    def flaky(src, dst):
        state["calls"] += 1
        if state["n"] > 0 and str(dst).endswith(suffix):
            state["n"] -= 1
            raise PermissionError(32, "sharing violation")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return state


# --- G1: save_bytes_atomic contract -------------------------------------------------------------------------------


def test_g1a_bytes_are_written_exactly(tmp_path):
    p = tmp_path / "k.bin"
    save_bytes_atomic(p, b"\x00\x01\xff raw")
    assert p.read_bytes() == b"\x00\x01\xff raw"
    save_bytes_atomic(p, b"second")
    assert p.read_bytes() == b"second"


def test_g1b_two_saves_use_two_unique_temp_names_of_the_shared_shape(tmp_path, monkeypatch):
    seen: list[str] = []
    real = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    p = tmp_path / "k.bin"
    save_bytes_atomic(p, b"a")
    save_bytes_atomic(p, b"b")
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(TEMP_RE.search(n) and n.startswith("k.bin.") for n in seen), seen


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_g1c_mode_is_applied_to_the_temp_before_it_is_published(tmp_path, monkeypatch):
    modes: list[int] = []
    real = os.replace

    def spy(src, dst):
        modes.append(stat.S_IMODE(os.stat(src).st_mode))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    p = tmp_path / "k.bin"
    save_bytes_atomic(p, b"secret", mode=0o600)
    assert modes == [0o600]
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_g1d_fsync_order_file_then_replace_then_directory(tmp_path, monkeypatch):
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("dir-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync")
        return real_fsync(fd)

    def replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    save_bytes_atomic(tmp_path / "k.bin", b"x")
    if os.name == "posix":
        assert events == ["file-fsync", "replace", "dir-fsync"]
    else:
        assert events == ["file-fsync", "replace"]
    events.clear()
    save_bytes_atomic(tmp_path / "k2.bin", b"x", fsync=False)
    assert events == ["replace"]


def test_g1e_transient_permission_errors_are_retried_and_a_persistent_one_is_raised(tmp_path, monkeypatch):
    p = tmp_path / "k.bin"
    state = _flaky_replace(monkeypatch, "k.bin", 2)
    save_bytes_atomic(p, b"ok")
    assert p.read_bytes() == b"ok" and state["n"] == 0

    p2 = tmp_path / "persist.bin"
    p2.write_bytes(b"old")
    state = _flaky_replace(monkeypatch, "persist.bin", 10_000)
    with pytest.raises(PermissionError):
        save_bytes_atomic(p2, b"new")
    assert state["calls"] == aw._REPLACE_RETRIES
    assert p2.read_bytes() == b"old"
    assert _temps(tmp_path) == []


@pytest.mark.parametrize("exc", [OSError("disk full"), KeyboardInterrupt(), SystemExit(1)])
def test_g1f_failure_during_the_write_leaves_destination_unchanged_and_no_temp(tmp_path, monkeypatch, exc):
    p = tmp_path / "k.bin"
    p.write_bytes(b"old")
    absent = tmp_path / "absent.bin"

    def boom(fd):
        raise exc

    monkeypatch.setattr(os, "fsync", boom)
    for target in (p, absent):
        with pytest.raises(type(exc)):
            save_bytes_atomic(target, b"new")
    assert p.read_bytes() == b"old" and not absent.exists()
    assert _temps(tmp_path) == []


def test_g1f_a_kill_stand_in_runs_no_cleanup_and_the_next_save_sweeps_the_leftover(tmp_path, monkeypatch):
    p = tmp_path / "k.bin"
    p.write_bytes(b"old")

    def boom(fd):
        raise SimulatedKill()

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(SimulatedKill):
        save_bytes_atomic(p, b"new")
    assert p.read_bytes() == b"old"
    assert len(_temps(tmp_path)) == 1  # a real kill leaves its temp; nothing ran to remove it
    monkeypatch.undo()
    save_bytes_atomic(p, b"newer")
    assert p.read_bytes() == b"newer" and _temps(tmp_path) == []


def test_g1g_a_dead_writers_temp_is_swept_and_a_live_other_process_temp_is_left(tmp_path):
    p = tmp_path / "k.bin"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        stale = tmp_path / f"k.bin.t{dead.pid}-aaaaaaaaaaaa.new"
        live = tmp_path / f"k.bin.t{sleeper.pid}-bbbbbbbbbbbb.new"
        stale.write_bytes(b"x")
        live.write_bytes(b"y")
        save_bytes_atomic(p, b"data")
        assert not stale.exists()
        assert live.exists()
    finally:
        sleeper.kill()
        sleeper.wait()


# --- G2 (new): the text writer cleans up on interrupts ------------------------------------------------------------


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1)])
def test_g2_text_writer_removes_its_temp_on_keyboard_interrupt_and_system_exit(tmp_path, monkeypatch, exc):
    p = tmp_path / "t.txt"
    p.write_text("old", encoding="utf-8")

    def boom(fd):
        raise exc

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(type(exc)):
        save_with_backup_text(p, "new", keep_backups=False)
    assert p.read_text(encoding="utf-8") == "old"
    assert _temps(tmp_path) == []


# --- G3: fsync_dir ------------------------------------------------------------------------------------------------


def test_g3c_fsync_dir_never_raises_and_is_posix_only(tmp_path, monkeypatch):
    fsync_dir(tmp_path / "does-not-exist")  # no error
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    fsync_dir(tmp_path)
    assert len(calls) == (1 if os.name == "posix" else 0)
    calls.clear()
    monkeypatch.setattr(aw.os, "name", "nt")
    fsync_dir(tmp_path)
    assert calls == []


def test_g3a_the_save_functions_call_fsync_dir_when_fsync_is_on(tmp_path, monkeypatch):
    seen: list[Path] = []
    monkeypatch.setattr(aw, "fsync_dir", lambda d: seen.append(Path(d)))
    save_bytes_atomic(tmp_path / "a.bin", b"x")
    save_with_backup_text(tmp_path / "b.txt", "x", keep_backups=False)
    assert seen == [tmp_path, tmp_path]
    seen.clear()
    save_bytes_atomic(tmp_path / "c.bin", b"x", fsync=False)
    save_with_backup_text(tmp_path / "d.txt", "x", keep_backups=False, fsync=False)
    assert seen == []


# --- G4: the blob writer ------------------------------------------------------------------------------------------


def test_g4_blob_written_once_then_deduped_with_no_write_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "sub" / "abc"
    save_blob_once(target, b"data")
    assert target.read_bytes() == b"data" and _temps(target.parent) == []
    calls: list[Path] = []
    real = aw.save_bytes_atomic
    monkeypatch.setattr(aw, "save_bytes_atomic", lambda p, d, **k: (calls.append(p), real(p, d, **k))[1])
    save_blob_once(target, b"data")
    assert calls == []


def test_g4d_blob_failure_with_the_target_absent_propagates_and_leaves_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "abc"
    real = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save_blob_once(target, b"data")
    monkeypatch.setattr(os, "replace", real)
    assert not target.exists() and _temps(tmp_path) == []


def test_g4e_blob_failure_with_the_target_present_means_a_concurrent_writer_won(tmp_path, monkeypatch):
    target = tmp_path / "abc"
    real = os.replace
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def competitor_then_denied(src, dst):
        if not target.exists():
            target.write_bytes(b"data")  # a competing writer lands the same content first
        raise PermissionError(32, "held open")

    monkeypatch.setattr(os, "replace", competitor_then_denied)
    save_blob_once(target, b"data")  # no error: the target exists with the same content
    monkeypatch.setattr(os, "replace", real)
    assert target.read_bytes() == b"data" and _temps(tmp_path) == []


# --- G16: sweep and in-flight interleavings -----------------------------------------------------------------------


def test_g16a_threads_writing_one_blob_end_with_one_file_and_no_residue(tmp_path):
    target = tmp_path / "blobs" / "same"
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def work():
        try:
            barrier.wait()
            save_blob_once(target, b"identical bytes")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert [p.name for p in target.parent.iterdir()] == ["same"]
    assert target.read_bytes() == b"identical bytes"


def test_g16a_two_processes_writing_one_blob_end_with_one_file(tmp_path):
    target = tmp_path / "blobs" / "same"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from pathlib import Path\n"
        "from brain.utils.atomic_write import save_blob_once\n"
        f"save_blob_once(Path({str(target)!r}), b'identical bytes')\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(2)]
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    assert [p.name for p in target.parent.iterdir()] == ["same"]
    assert target.read_bytes() == b"identical bytes"


def test_g16c_a_sweep_during_a_save_does_not_remove_that_saves_own_temp(tmp_path, monkeypatch):
    p = tmp_path / "k.bin"
    real = os.replace

    def sweep_then_replace(src, dst):
        aw._remove_stale_temps(p)  # a concurrent saver's sweep, run while this save is in flight
        return real(src, dst)

    monkeypatch.setattr(os, "replace", sweep_then_replace)
    save_bytes_atomic(p, b"data")
    assert p.read_bytes() == b"data"


def test_g16c_the_oracle_can_fail_when_the_in_flight_registry_is_ignored(tmp_path, monkeypatch):
    p = tmp_path / "k.bin"
    real = os.replace

    def sweep_then_replace(src, dst):
        aw._remove_stale_temps(p)
        return real(src, dst)

    monkeypatch.setattr(os, "replace", sweep_then_replace)
    monkeypatch.setattr(aw, "_INFLIGHT", _EverythingIsStale())
    with pytest.raises(FileNotFoundError):
        save_bytes_atomic(p, b"data")


class _EverythingIsStale(set):
    def __contains__(self, item):
        return False

    def update(self, *a, **k):  # registration is a no-op: the in-flight guard is ignored
        return None

    def difference_update(self, *a, **k):
        return None


# --- G20: placement: one module, no shim, no cycle ----------------------------------------------------------------

NEW = REPO / "brain" / "utils" / "atomic_write.py"
HEAL = REPO / "brain" / "health" / "attempt_heal.py"
HEAL_MOD = "brain.health.attempt_heal"


def _moved_names(path: Path = NEW) -> set[str]:
    """Top-level names defined or assigned in the new module (not merely imported)."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _defined_or_assigned(path: Path) -> set[str]:
    return _moved_names(path)


def backdoor_findings(source: str, moved: set[str]) -> list[str]:
    """Ways a module can reach a moved writer through attempt_heal (the shim this change must not leave)."""
    out: list[str] = []
    tree = ast.parse(source)
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == HEAL_MOD:
            for a in node.names:
                if a.name == "*":
                    out.append("star import")
                elif a.name in moved:
                    out.append(f"imports {a.name}")
        elif isinstance(node, ast.ImportFrom) and node.module == "brain.health":
            aliases.update(a.asname or a.name for a in node.names if a.name == "attempt_heal")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == HEAL_MOD:
                    aliases.add(a.asname or HEAL_MOD)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in moved:
            if ast.unparse(node.value) in aliases:
                out.append(f"attribute {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(node.value == f"{HEAL_MOD}.{n}" for n in moved):
                out.append(f"string {node.value}")
    return out


def test_g20a_the_new_module_holds_the_writers_and_attempt_heal_holds_none_of_them():
    moved = _moved_names()
    for name in (
        "save_with_backup_text",
        "save_bytes_atomic",
        "fsync_dir",
        "save_blob_once",
        "_save_payload",
        "_temp_scope",
        "_remove_stale_temps",
        "_owner_alive",
        "_copy_for_backup",
        "_retry_perm",
        "_replace_with_retry",
        "_REPLACE_RETRIES",
        "_REPLACE_RETRY_SLEEP_S",
        "_INFLIGHT",
        "_INFLIGHT_LOCK",
    ):
        assert name in moved, name
    heal_defs = _defined_or_assigned(HEAL)
    assert not (heal_defs & moved), heal_defs & moved
    assert {
        "save_with_backup",
        "attempt_heal_text",
        "_heal_text_from_baks",
        "_classify_cause",
        "_filename_safe_timestamp",
    } <= heal_defs
    import brain.health.attempt_heal as heal

    for name in moved - {"save_with_backup_text"}:
        assert not hasattr(heal, name), name
    tree = ast.parse(HEAL.read_text(encoding="utf-8"))
    imports_from_new = [
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "brain.utils.atomic_write"
        for a in n.names
    ]
    assert imports_from_new == ["save_with_backup_text"]
    assert "__all__" not in heal_defs
    assert not any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree))


def _py_files():
    for root in ("brain", "tests", "scripts"):
        yield from sorted((REPO / root).rglob("*.py"))


def test_g20b_no_module_reaches_a_moved_writer_through_attempt_heal():
    moved = _moved_names()
    offenders = {}
    for path in _py_files():
        if path == Path(__file__).resolve():
            continue  # this file names the offenders in its own table below
        found = backdoor_findings(path.read_text(encoding="utf-8"), moved)
        if found:
            offenders[path.relative_to(REPO).as_posix()] = found
    assert offenders == {}


def test_g20b_the_scan_flags_each_kind_of_offender():
    moved = {"save_with_backup_text", "_copy_for_backup", "_INFLIGHT"}
    cases = {
        "from brain.health.attempt_heal import _copy_for_backup": "imports _copy_for_backup",
        "from brain.health.attempt_heal import *": "star import",
        "import brain.health.attempt_heal as heal\nheal.save_with_backup_text(p, t)": "attribute save_with_backup_text",
        "from brain.health import attempt_heal\nattempt_heal._INFLIGHT.clear()": "attribute _INFLIGHT",
        "def f():\n    import brain.health.attempt_heal as h\n    return h.save_with_backup_text": "attribute save_with_backup_text",
        'monkeypatch.setattr("brain.health.attempt_heal._copy_for_backup", f)': "string brain.health.attempt_heal._copy_for_backup",
    }
    for src, expected in cases.items():
        assert expected in backdoor_findings(src, moved), src
    assert backdoor_findings("from brain.health.attempt_heal import save_with_backup", moved) == []
    assert backdoor_findings("from brain.utils.atomic_write import _copy_for_backup", moved) == []


def _fresh(code: str, repo: Path = REPO) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with `repo` first on sys.path (the venv's `brain` is an editable install of
    another checkout, and -I would ignore PYTHONPATH, so the path is inserted explicitly)."""
    prelude = f"import sys\nsys.path.insert(0, {str(repo)!r})\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + code], capture_output=True, text=True, cwd=str(repo), timeout=120
    )


def test_g20c_importing_the_new_module_pulls_in_nothing_from_health():
    out = _fresh(
        "import brain.utils.atomic_write as m\n"
        "import os\n"
        f"assert os.path.realpath(m.__file__).startswith(os.path.realpath({str(REPO)!r})), m.__file__\n"
        "print(sorted(x for x in sys.modules if x == 'brain.health' or x.startswith('brain.health.')))\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("module", ["brain.health.json_io", "brain.health.attempt_heal", "brain.health.log_rotation"])
def test_g20c_each_health_module_imports_cleanly_in_a_fresh_interpreter(module):
    out = _fresh(
        f"import {module} as m\n"
        "import os\n"
        f"assert os.path.realpath(m.__file__).startswith(os.path.realpath({str(REPO)!r})), m.__file__\n"
    )
    assert out.returncode == 0, out.stderr


def test_g20c_the_check_can_fail_with_the_wrong_tree_or_a_cycle(tmp_path):
    wrong = _fresh("import brain\nimport os\nassert os.path.realpath(brain.__file__).startswith('/nonexistent')\n")
    assert wrong.returncode != 0  # the __file__ assertion is real
    fake = tmp_path / "brain" / "utils"
    fake.mkdir(parents=True)
    (tmp_path / "brain" / "__init__.py").write_text("")
    (fake / "__init__.py").write_text("")
    (tmp_path / "brain" / "health").mkdir()
    (tmp_path / "brain" / "health" / "__init__.py").write_text("")
    (tmp_path / "brain" / "health" / "anomaly.py").write_text("")
    (fake / "atomic_write.py").write_text("import brain.health.anomaly\n")
    out = _fresh(
        "import brain.utils.atomic_write\n"
        "print(sorted(x for x in sys.modules if x == 'brain.health' or x.startswith('brain.health.')))\n",
        repo=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() != "[]"  # the sys.modules assertion flags the stand-in


def test_g20d_every_importer_takes_its_writers_from_the_new_module():
    importers = {
        "brain/migrator/cli.py": "save_with_backup_text",
        "brain/bridge/server.py": "save_with_backup_text",
        "brain/works/storage.py": "save_with_backup_text",
        "brain/engines/research_notes.py": "save_with_backup_text",
        "brain/ingest/buffer.py": "save_with_backup_text",
        "brain/service/systemd.py": "save_with_backup_text",
        "brain/service/windows_service.py": "save_with_backup_text",
        "brain/health/json_io.py": "save_with_backup_text",
        "brain/health/log_rotation.py": "_replace_with_retry",
    }
    for rel, name in importers.items():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        got = {
            a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module == "brain.utils.atomic_write"
            for a in n.names
        }
        assert name in got, rel
