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


def test_a_handle_that_resolves_into_a_protected_folder_is_refused_and_the_file_removed(env, monkeypatch):
    # Windows path, any OS: the open is redirected into ~/.ssh as a junction would, and the handle's
    # real location (not the requested path) decides; the file it created there is removed again
    from brain.files import commit as commit_mod

    target, landed = env["home"] / "Notes" / "rc", env["home"] / ".ssh" / "rc"
    rid = propose_write(path=str(target), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_open_final = commit_mod._open_final
    monkeypatch.setattr(commit_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(commit_mod, "_open_final", lambda name, op, dir_fd: real_open_final(str(landed), op, dir_fd))
    monkeypatch.setattr(commit_mod, "_handle_path", lambda fd: str(landed))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not landed.exists() and not target.exists()
    assert pending.get(env["persona"], rid)["status"] == "refused"


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
