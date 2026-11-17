"""#699 on Windows: no dir_fd or O_NOFOLLOW there, so commit opens by path and then
judges where the opened handle really points (a junction swapped in is resolved by
the handle). The decision is tested on every OS through the platform seam; the real
junction swap only runs on Windows. Temp HOME and KINDLED_HOME; no mocked state files.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from brain.files import pending
from brain.tools.impls.propose_write import propose_write


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    kh = tmp_path / "kindled"
    persona = kh / "personas" / "canary"
    for d in (home / ".ssh", home / "Notes", persona):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("KINDLED_HOME", str(kh))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: home)
    return {"home": home, "persona": persona}


def _commit(env, rid):
    from brain.files.commit import commit_write
    from brain.memory.store import MemoryStore

    store = MemoryStore(env["persona"] / "memories.db")
    try:
        return commit_write(env["persona"], rid, store=store)
    finally:
        store.close()


@pytest.mark.parametrize("into", [".ssh", "Other"])
def test_a_handle_that_resolves_anywhere_but_the_approved_path_is_refused_and_the_file_removed(env, monkeypatch, into):
    # Windows path, any OS: the open is redirected as a junction would (into a protected folder, or just
    # somewhere else), and the handle's real location decides; the file it created there is removed again
    from brain.files import commit as commit_mod

    (env["home"] / into).mkdir(exist_ok=True)
    target, landed = env["home"] / "Notes" / "rc", env["home"] / into / "rc"
    rid = propose_write(path=str(target), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_open_final = commit_mod._open_final
    monkeypatch.setattr("brain.files.write_guard._is_windows", lambda: True)
    monkeypatch.setattr(commit_mod, "_open_final", lambda name, op, dir_fd: real_open_final(str(landed), op, dir_fd))
    monkeypatch.setattr(commit_mod, "_handle_path", lambda fd: str(landed))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not landed.exists() and not target.exists()
    assert pending.get(env["persona"], rid)["status"] == "refused"


def _last_audit_event(env):
    import json

    log = env["persona"] / "write_audit.jsonl"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()][-1]["event"]


@pytest.mark.parametrize("where", ["at_the_recorded_path", "under_the_handles_real_location"])
def test_a_refused_create_on_windows_removes_the_folders_it_made_703(env, monkeypatch, where):
    # Windows path, any OS (#703): the missing folders were made by path and left after the handle check
    # refused. They are removed where they are: at the path they were made at, or, when the folders were moved
    # (a junction swapped in), under the location the handle really reports.
    from brain.files import commit as commit_mod

    notes, ssh, other = env["home"] / "Notes", env["home"] / ".ssh", env["home"] / "Other"
    other.mkdir()
    target = notes / "a" / "b" / "rc"
    rid = propose_write(path=str(target), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    landed = ssh / "rc" if where == "at_the_recorded_path" else other / "a" / "b" / "rc"
    real_open_final = commit_mod._open_final

    def redirected_open(name, op, dir_fd):
        if where == "under_the_handles_real_location":
            (notes / "a").rename(other / "a")  # the folders now really sit under Other
        return real_open_final(str(landed), op, dir_fd)

    monkeypatch.setattr("brain.files.write_guard._is_windows", lambda: True)
    monkeypatch.setattr(commit_mod, "_open_final", redirected_open)
    monkeypatch.setattr(commit_mod, "_handle_path", lambda fd: str(landed))
    done = _commit(env, rid)
    assert done["ok"] is False and done["error"] == f"refused: {target} leads to {landed} now", done
    assert pending.get(env["persona"], rid)["status"] == "refused"
    assert _last_audit_event(env) == "commit_refused"
    assert list(notes.iterdir()) == [] and list(other.iterdir()) == []
    assert sorted(p.name for p in ssh.iterdir()) == []


@pytest.mark.no_base_failure(reason="#703 identity rule on the Windows path: the cleanup must not remove what this call did not make, or what it cannot identify; the base removes nothing. Shown able to fail by mutations recorded in the run's harness")
@pytest.mark.parametrize("case", ["made_by_another_writer", "same_name_swapped_in", "no_file_index"])
def test_the_windows_cleanup_removes_only_folders_it_can_identify_as_its_own_703(env, monkeypatch, case):
    import os

    from brain.files import commit as commit_mod

    notes, ssh = env["home"] / "Notes", env["home"] / ".ssh"
    a, b = notes / "a", notes / "a" / "b"
    rid = propose_write(path=str(b / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_mkdir, real_stat, real_open_final = os.mkdir, os.stat, commit_mod._open_final
    if case == "made_by_another_writer":  # our mkdir of 'a' then raises FileExistsError: 'a' is not ours
        def mkdir(path, mode=0o777, *, dir_fd=None):
            if dir_fd is None and Path(path) == a and not a.exists():
                real_mkdir(path, mode)
            return real_mkdir(path, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "mkdir", mkdir)
    if case == "no_file_index":  # a volume with no file index reports st_ino 0
        def stat(path, *args, **kwargs):
            st = real_stat(path, *args, **kwargs)
            if isinstance(path, (str, Path)) and str(path).startswith(str(notes)):
                fields = list(st)[:10]
                fields[1] = 0
                return os.stat_result(fields)
            return st

        monkeypatch.setattr(os, "stat", stat)

    def redirected_open(name, op, dir_fd):
        if case == "same_name_swapped_in":
            b.rename(a / "b.ours")
            b.mkdir()  # someone else's empty folder under our folder's name
        return real_open_final(str(ssh / "rc"), op, dir_fd)

    monkeypatch.setattr("brain.files.write_guard._is_windows", lambda: True)
    monkeypatch.setattr(commit_mod, "_open_final", redirected_open)
    monkeypatch.setattr(commit_mod, "_handle_path", lambda fd: str(ssh / "rc"))
    assert _commit(env, rid)["ok"] is False
    if case == "made_by_another_writer":
        assert a.is_dir() and list(a.iterdir()) == []  # theirs kept; ours ('b') removed
    elif case == "same_name_swapped_in":
        assert b.is_dir() and list(b.iterdir()) == []  # theirs kept (ours, moved, stays: accepted on #703)
    else:
        assert b.is_dir() and a.is_dir()  # no identity, no removal


def test_a_handle_at_a_drive_root_still_refuses_cleanly_and_removes_the_folders_703(env, monkeypatch):
    # A junction to a drive or share root makes the real location shallower than the folders made for the file:
    # looking for them under it must not raise past the refusal (the record would stay 'committing', with no
    # refusal recorded and an unhandled error at the approval).
    from brain.files import commit as commit_mod

    notes, ssh = env["home"] / "Notes", env["home"] / ".ssh"
    rid = propose_write(path=str(notes / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_open_final = commit_mod._open_final
    monkeypatch.setattr("brain.files.write_guard._is_windows", lambda: True)
    monkeypatch.setattr(commit_mod, "_open_final", lambda name, op, dir_fd: real_open_final(str(ssh / "rc"), op, dir_fd))
    shallow = str(Path(Path.cwd().anchor) / "rc")
    monkeypatch.setattr(commit_mod, "_handle_path", lambda fd: shallow)
    import os

    real_unlink = os.unlink  # the refusal removes the file at the real location: keep that inside this test's tmp
    monkeypatch.setattr(os, "unlink", lambda p, *a, **k: real_unlink(p, *a, **k) if str(p).startswith(str(env["home"])) else None)
    done = _commit(env, rid)
    assert done == {"ok": False, "error": f"refused: {notes / 'a' / 'b' / 'rc'} leads to {shallow} now"}
    assert pending.get(env["persona"], rid)["status"] == "refused"
    assert _last_audit_event(env) == "commit_refused"
    assert list(notes.iterdir()) == []


@pytest.mark.timeout(5)
def test_the_walk_up_to_the_first_existing_folder_stops_at_a_missing_root_703():
    # #703: Windows makes each missing folder itself now. Under a drive that is gone (a card can wait 24 h),
    # a walk up while the folder does not exist must end at the root, whose parent is itself, not loop.
    from pathlib import PureWindowsPath

    from brain.files.commit import _missing_folders

    missing = _missing_folders(PureWindowsPath("Z:\\a\\b"), exists=lambda p: False)
    assert missing == [PureWindowsPath("Z:\\a"), PureWindowsPath("Z:\\a\\b")]
    assert _missing_folders(PureWindowsPath("Z:\\a\\b"), exists=lambda p: str(p) in ("Z:\\", "Z:\\a")) == [PureWindowsPath("Z:\\a\\b")]


@pytest.mark.no_base_failure(reason="Windows-only (junctions); the base-code check runs on Linux. The seam test above bites there")
@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_a_folder_swapped_for_a_junction_is_refused_on_windows(env, monkeypatch):
    from brain.files import commit as commit_mod

    notes = env["home"] / "Notes"
    rid = propose_write(path=str(notes / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_mark = commit_mod.pending.mark

    def mark(persona_dir, record_id, *, status, **kw):
        ok = real_mark(persona_dir, record_id, status=status, **kw)
        if status == "committing":
            notes.rename(notes.with_name("Notes.moved"))
            subprocess.run(["cmd", "/c", "mklink", "/J", str(notes), str(env["home"] / ".ssh")], check=True, capture_output=True)
        return ok

    monkeypatch.setattr(commit_mod.pending, "mark", mark)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not (env["home"] / ".ssh" / "rc").exists()


@pytest.mark.no_base_failure(reason="Windows-only (junctions); the base-code check runs on Linux. Shown red on windows-latest before the #703 fix")
@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_a_refused_create_through_a_junction_leaves_no_folder_in_its_target_703(env, monkeypatch):
    # #703: the missing folders were made by path, so through the junction, inside ~/.ssh; the refusal
    # removed the file but left them there
    from brain.files import commit as commit_mod

    notes, ssh = env["home"] / "Notes", env["home"] / ".ssh"
    before = sorted(p.name for p in ssh.iterdir())
    rid = propose_write(path=str(notes / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_mark = commit_mod.pending.mark

    def mark(persona_dir, record_id, *, status, **kw):
        ok = real_mark(persona_dir, record_id, status=status, **kw)
        if status == "committing":
            notes.rename(notes.with_name("Notes.moved"))
            subprocess.run(["cmd", "/c", "mklink", "/J", str(notes), str(ssh)], check=True, capture_output=True)
        return ok

    monkeypatch.setattr(commit_mod.pending, "mark", mark)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert sorted(p.name for p in ssh.iterdir()) == before
