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
