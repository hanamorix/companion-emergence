"""#699: commit creates the file inside the folder it checked, not one looked up again by path.

The guard checks a path, then commit used to resolve the folders again by path, so
a folder swapped for a symlink in between was followed: a create approved for
``~/Notes/rc`` landed as ``~/.ssh/rc`` (which sshd runs at login). The swap is made
at the claim step, between commit's guard re-run and its open. Temp HOME and
KINDLED_HOME; no mocked state files.
"""
import sys
from pathlib import Path

import pytest

from brain.files import pending
from brain.tools.impls.propose_write import propose_write

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink swaps; the Windows junction case is its own test")


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
    (home / ".ssh" / "authorized_keys").write_text("original\n", encoding="utf-8")
    return {"home": home, "persona": persona}


def _at_claim(monkeypatch, swap):
    """Run `swap()` when commit claims the card: after its guard re-run, before it opens anything."""
    from brain.files import commit as commit_mod

    real_mark = commit_mod.pending.mark

    def mark(persona_dir, rid, *, status, **kw):
        ok = real_mark(persona_dir, rid, status=status, **kw)
        if status == "committing":
            swap()
        return ok

    monkeypatch.setattr(commit_mod.pending, "mark", mark)


def _commit(env, rid):
    from brain.files.commit import commit_write
    from brain.memory.store import MemoryStore

    store = MemoryStore(env["persona"] / "memories.db")
    try:
        return commit_write(env["persona"], rid, store=store)
    finally:
        store.close()


def _swap_folder_for_link(folder: Path, to: Path):
    def swap():
        folder.rename(folder.with_name(folder.name + ".moved"))
        folder.symlink_to(to, target_is_directory=True)
    return swap


def test_a_create_does_not_follow_a_parent_folder_swapped_for_a_link(env, monkeypatch):
    rid = propose_write(path=str(env["home"] / "Notes" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _at_claim(monkeypatch, _swap_folder_for_link(env["home"] / "Notes", env["home"] / ".ssh"))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not (env["home"] / ".ssh" / "rc").exists()
    assert pending.get(env["persona"], rid)["status"] == "refused"


def test_a_folder_swapped_for_home_cannot_turn_an_allowed_name_into_a_login_file(env, monkeypatch):
    # ~/Notes/.zshrc is allowed; with ~/Notes swapped for a link to ~ it would create ~/.zshrc
    rid = propose_write(path=str(env["home"] / "Notes" / ".zshrc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _at_claim(monkeypatch, _swap_folder_for_link(env["home"] / "Notes", env["home"]))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not (env["home"] / ".zshrc").exists()


def test_missing_folders_are_not_made_inside_a_swapped_parent(env, monkeypatch):
    rid = propose_write(path=str(env["home"] / "Notes" / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _at_claim(monkeypatch, _swap_folder_for_link(env["home"] / "Notes", env["home"] / ".ssh"))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert sorted(p.name for p in (env["home"] / ".ssh").iterdir()) == ["authorized_keys"]


def test_a_link_planted_where_a_missing_folder_goes_is_not_followed(env, monkeypatch):
    target = env["home"] / "Notes" / "new" / "rc"
    rid = propose_write(path=str(target), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _at_claim(monkeypatch, lambda: target.parent.symlink_to(env["home"] / ".ssh", target_is_directory=True))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not (env["home"] / ".ssh" / "rc").exists()
    assert pending.get(env["persona"], rid)["status"] == "refused"


def test_a_folder_moved_into_a_protected_one_while_the_file_opens_is_caught_and_undone(env, monkeypatch):
    # the pinned folder itself is renamed into ~/.ssh as the file is created: the second check sees it
    from brain.files import commit as commit_mod

    rid = propose_write(path=str(env["home"] / "Notes" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    real_open_final = commit_mod._open_final

    def open_then_move(name, op, dir_fd):
        fd = real_open_final(name, op, dir_fd)
        (env["home"] / "Notes").rename(env["home"] / ".ssh" / "Notes")
        return fd

    monkeypatch.setattr(commit_mod, "_open_final", open_then_move)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not (env["home"] / ".ssh" / "Notes" / "rc").exists()
    assert pending.get(env["persona"], rid)["status"] == "refused"


def _last_audit_event(env):
    import json

    log = env["persona"] / "write_audit.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1]["event"]


@pytest.mark.parametrize("moved", ["b", "a", "Notes"])
def test_a_refused_create_leaves_no_folder_it_made_wherever_it_was_moved_703(env, monkeypatch, moved):
    # #703: the file was removed after the post-open check refused, the folders this call made were not.
    # The racer moves one folder on the path into ~/.ssh while the file opens: one we made (b or a), or
    # Notes, which existed before (its move stays: it is the racer's own act on a folder we did not make).
    from brain.files import commit as commit_mod

    home, notes, ssh = env["home"], env["home"] / "Notes", env["home"] / ".ssh"
    rid = propose_write(path=str(notes / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    source = {"b": notes / "a" / "b", "a": notes / "a", "Notes": notes}[moved]
    real_open_final = commit_mod._open_final

    def open_then_move(name, op, dir_fd):
        fd = real_open_final(name, op, dir_fd)
        source.rename(ssh / source.name)
        return fd

    monkeypatch.setattr(commit_mod, "_open_final", open_then_move)
    done = _commit(env, rid)
    assert done == {"ok": False, "error": f"refused: {notes / 'a' / 'b' / 'rc'} moved into a protected location while it was opened"}
    assert pending.get(env["persona"], rid)["status"] == "refused"
    assert _last_audit_event(env) == "commit_refused"
    if moved == "Notes":
        assert sorted(p.name for p in ssh.iterdir()) == ["Notes", "authorized_keys"]
        assert list((ssh / "Notes").iterdir()) == []
    else:
        assert sorted(p.name for p in ssh.iterdir()) == ["authorized_keys"]
        assert list(notes.iterdir()) == []
    assert not (home / "a").exists()


def test_a_made_folder_is_found_when_its_parent_was_not_ours_and_was_moved_703(env, monkeypatch):
    # Another writer made 'a', so it is not ours and its descriptor is not held; ours is 'b' inside it. The
    # racer moves 'a' into ~/.ssh: 'b' can no longer be reached by name from Notes, but its own descriptor finds it.
    import os

    from brain.files import commit as commit_mod

    notes, ssh = env["home"] / "Notes", env["home"] / ".ssh"
    rid = propose_write(path=str(notes / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _plant_before_mkdir(monkeypatch, "a", lambda dir_fd: os.mkdir("a", dir_fd=dir_fd))
    real_open_final = commit_mod._open_final

    def open_then_move(name, op, dir_fd):
        fd = real_open_final(name, op, dir_fd)
        (notes / "a").rename(ssh / "a")
        return fd

    monkeypatch.setattr(commit_mod, "_open_final", open_then_move)
    done = _commit(env, rid)
    assert done == {"ok": False, "error": f"refused: {notes / 'a' / 'b' / 'rc'} moved into a protected location while it was opened"}
    assert pending.get(env["persona"], rid)["status"] == "refused"
    assert _last_audit_event(env) == "commit_refused"
    assert list(notes.iterdir()) == []
    assert (ssh / "a").is_dir() and list((ssh / "a").iterdir()) == []  # theirs stays where the racer put it; ours is gone


@pytest.mark.parametrize("raised, status", [(PermissionError(13, "Permission denied"), "error"), ("changed", "refused")])
def test_a_create_whose_open_fails_after_making_folders_leaves_none_of_them_703(env, monkeypatch, raised, status):
    # #703: an OSError at the open (P3), and the refusal type itself, left the folders made for it
    from brain.files import commit as commit_mod

    notes = env["home"] / "Notes"
    rid = propose_write(path=str(notes / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    exc = commit_mod._TargetChangedError("refused: rc appeared after it was checked") if raised == "changed" else raised

    def failing_open(name, op, dir_fd):
        raise exc

    monkeypatch.setattr(commit_mod, "_open_final", failing_open)
    done = _commit(env, rid)
    assert done["ok"] is False and done["error"] == str(exc), done
    assert pending.get(env["persona"], rid)["status"] == status
    assert _last_audit_event(env) == {"error": "error", "refused": "commit_refused"}[status]
    assert list(notes.iterdir()) == []


def _plant_before_mkdir(monkeypatch, name, plant):
    """Run `plant(dir_fd)` just before commit's ``os.mkdir(name, dir_fd=...)``, then the real mkdir."""
    import os

    real_mkdir = os.mkdir
    planting = []

    def mkdir(path, mode=0o777, *, dir_fd=None):
        if path == name and dir_fd is not None and not planting:
            planting.append(True)  # the racer's own os.mkdir must not plant again
            try:
                plant(dir_fd)
            finally:
                planting.clear()
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)


@pytest.mark.no_base_failure(reason="#703 ratified residue: a made folder holding a racer's own entry stays; the base leaves it too. Shown able to fail with an rmtree cleanup")
def test_a_made_folder_holding_a_racers_link_stays_with_the_link_703(env, monkeypatch):
    # The racer plants a link where the next folder goes: refused (#699). Our 'new' now holds the racer's
    # link, so it stays: the cleanup never deletes an entry it did not make (owner decision on #703).
    import os

    notes, ssh = env["home"] / "Notes", env["home"] / ".ssh"
    rid = propose_write(path=str(notes / "new" / "sub" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _plant_before_mkdir(monkeypatch, "sub", lambda dir_fd: os.symlink(str(ssh), "sub", dir_fd=dir_fd))
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert pending.get(env["persona"], rid)["status"] == "refused"
    assert [p.name for p in (notes / "new").iterdir()] == ["sub"] and (notes / "new" / "sub").is_symlink()
    assert sorted(p.name for p in ssh.iterdir()) == ["authorized_keys"]
    assert (ssh / "authorized_keys").read_text(encoding="utf-8") == "original\n"


@pytest.mark.no_base_failure(reason="#703 identity rule: the cleanup must not remove what this call did not make; the base removes nothing. Shown able to fail by mutations recorded in the run's harness")
@pytest.mark.parametrize("case", ["made_by_another_writer", "same_name_swapped_in", "holds_a_file"])
def test_the_cleanup_removes_only_empty_folders_this_call_made_703(env, monkeypatch, case):
    import os

    from brain.files import commit as commit_mod

    notes = env["home"] / "Notes"
    a, b = notes / "a", notes / "a" / "b"
    rid = propose_write(path=str(b / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    if case == "made_by_another_writer":  # our mkdir of 'a' then raises FileExistsError: 'a' is not ours
        _plant_before_mkdir(monkeypatch, "a", lambda dir_fd: os.mkdir("a", dir_fd=dir_fd))

    def failing_open(name, op, dir_fd):
        if case == "same_name_swapped_in":
            b.rename(a / "b.ours")
            b.mkdir()  # someone else's empty folder under our folder's name
        elif case == "holds_a_file":
            (b / "theirs.txt").write_text("keep me", encoding="utf-8")
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(commit_mod, "_open_final", failing_open)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    if case == "made_by_another_writer":
        assert a.is_dir() and list(a.iterdir()) == []  # theirs kept; ours ('b') removed
    elif case == "same_name_swapped_in":
        assert [p.name for p in a.iterdir()] == ["b"] and list(b.iterdir()) == []  # theirs kept, ours removed
    else:
        assert (b / "theirs.txt").read_text(encoding="utf-8") == "keep me"


@pytest.mark.no_base_failure(reason="#703: the cleanup follows no link; the base removes nothing. Shown able to fail by a cleanup that removes the resolved path")
def test_the_cleanup_does_not_follow_a_link_swapped_in_for_a_made_folder_703(env, monkeypatch):
    # The target is EMPTY, so an rmdir that followed the link would remove it: only a cleanup by descriptor
    # and identity keeps it.
    from brain.files import commit as commit_mod

    notes, other = env["home"] / "Notes", env["home"] / "Other"
    other.mkdir()
    a, b = notes / "a", notes / "a" / "b"
    rid = propose_write(path=str(b / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]

    def failing_open(name, op, dir_fd):
        b.rename(a / "b.ours")
        b.symlink_to(other, target_is_directory=True)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(commit_mod, "_open_final", failing_open)
    assert _commit(env, rid)["ok"] is False
    assert other.is_dir() and list(other.iterdir()) == []
    assert b.is_symlink() and [p.name for p in a.iterdir()] == ["b"]  # the link stays; ours (b.ours) is gone


@pytest.mark.skipif(hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0, reason="root ignores folder modes")
def test_a_refused_create_under_a_folder_you_can_enter_but_not_list_leaves_no_folder_703(env, monkeypatch):
    # Notes is enter-only (0o300). The made 'a' is removed by name in its parent's descriptor, which needs no
    # read permission; finding it through '..' and a listing would not work here.
    from brain.files import commit as commit_mod

    notes = env["home"] / "Notes"
    rid = propose_write(path=str(notes / "a" / "b" / "rc"), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    monkeypatch.setattr(commit_mod, "dir_fd_denied", lambda *a, **k: True)
    notes.chmod(0o300)
    try:
        done = _commit(env, rid)
    finally:
        notes.chmod(0o755)
    assert done["ok"] is False and pending.get(env["persona"], rid)["status"] == "refused", done
    assert list(notes.iterdir()) == []


def _open_fds() -> int:
    import os

    return len(os.listdir("/dev/fd"))


@pytest.mark.no_base_failure(reason="#703 holds descriptors for the cleanup; the base holds none past the call. Shown able to fail with one close dropped")
@pytest.mark.parametrize("case", ["no_leak_over_refusals", "deep_create_under_a_low_limit"])
def test_the_cleanup_leaks_no_descriptor_and_a_deep_create_still_succeeds_703(env, monkeypatch, case):
    import os

    from brain.files import commit as commit_mod

    notes = env["home"] / "Notes"
    if case == "no_leak_over_refusals":
        monkeypatch.setattr(commit_mod, "dir_fd_denied", lambda *a, **k: True)
        before = _open_fds()
        for i in range(200):
            with pytest.raises(commit_mod._TargetChangedError):
                commit_mod._open_pinned(notes / f"a{i}" / "b" / "rc", "create", env["persona"])
        assert _open_fds() == before
        assert list(notes.iterdir()) == []
    else:
        import resource

        target = notes.joinpath(*["d"] * 300) / "rc"
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        try:
            fd = commit_mod._open_pinned(target, "create", env["persona"])
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        os.close(fd)
        assert target.exists()


@pytest.mark.no_base_failure(reason="regression guard: creating missing parent folders already worked on the base; pins it against the #699 pinned mkdir")
def test_missing_parent_folders_are_still_created(env):
    target = env["home"] / "Notes" / "a" / "b" / "x.md"
    rid = propose_write(path=str(target), content="hello", op="create", persona_dir=env["persona"])["id"]
    assert _commit(env, rid)["ok"] is True
    assert target.read_text(encoding="utf-8") == "hello"


@pytest.mark.parametrize("where", ["missing_folder", "existing_folder"])
def test_a_link_to_an_unprotected_folder_elsewhere_is_not_followed_either(env, monkeypatch, where):
    # not a protected place, but not the one on the card: the file must land at the approved path or nowhere
    other = env["home"] / "Other"
    other.mkdir()
    if where == "missing_folder":
        target = env["home"] / "Notes" / "new" / "x.md"
        swap = lambda: target.parent.symlink_to(other, target_is_directory=True)  # noqa: E731
    else:
        target = env["home"] / "Notes" / "x.md"
        swap = _swap_folder_for_link(env["home"] / "Notes", other)
    rid = propose_write(path=str(target), content="PLANTED", op="create", persona_dir=env["persona"])["id"]
    _at_claim(monkeypatch, swap)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert list(other.iterdir()) == []
    assert pending.get(env["persona"], rid)["status"] == "refused"


@pytest.mark.parametrize("op", ["create", "append"])
def test_a_folder_swapped_while_the_card_waits_does_not_redirect_the_write(env, op):
    # the card names ~/Notes/x.md; the guard re-run resolved ~/Notes again and wrote wherever it now
    # pointed, so a swap before approval sent the write to an unlisted place (~/.config/fish/config.fish)
    other = env["home"] / "Other"
    other.mkdir()
    target = env["home"] / "Notes" / "x.md"
    if op == "append":
        target.write_text("mine\n", encoding="utf-8")
        (other / "x.md").write_text("other\n", encoding="utf-8")
    rid = propose_write(path=str(target), content="PLANTED", op=op, persona_dir=env["persona"])["id"]
    _swap_folder_for_link(env["home"] / "Notes", other)()  # no hook: the card is still waiting
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert "PLANTED" not in "".join(p.read_text(encoding="utf-8") for p in other.iterdir())
    assert pending.get(env["persona"], rid)["status"] == "refused"


def test_a_create_whose_folder_cannot_be_checked_after_the_open_is_refused_and_removed(env, monkeypatch):
    from brain.files import commit as commit_mod

    target = env["home"] / "Notes" / "x.md"
    rid = propose_write(path=str(target), content="PLANTED", op="create", persona_dir=env["persona"])["id"]

    def unreadable(*a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(commit_mod, "dir_fd_denied", unreadable)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert not target.exists()
    assert pending.get(env["persona"], rid)["status"] == "refused"
