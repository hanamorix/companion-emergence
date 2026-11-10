"""#694: an append to a file with more than one hard link is refused.

A hard link has no path back to the directory it came from, so neither the
spelling nor the identity comparison (#607) can tell that ``~/Notes/ak`` is
``~/.ssh/authorized_keys``. The guard can't see where the other links are, so
it refuses the append, and says why. Driven through the tools with a temp HOME
and KINDLED_HOME; no mocked state files.
"""
import os
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
    keys = home / ".ssh" / "authorized_keys"
    keys.write_text("ssh-ed25519 AAAA original\n", encoding="utf-8")
    return {"home": home, "persona": persona, "keys": keys}


def _link(src: Path, dst: Path) -> Path:
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hard links unsupported here: {exc}")
    return dst


def test_propose_append_to_a_hard_link_of_a_protected_file_is_refused(env):
    link = _link(env["keys"], env["home"] / "Notes" / "ak")
    out = propose_write(path=str(link), content="ssh-ed25519 AAAA planted", op="append", persona_dir=env["persona"])
    assert "hard link" in out.get("error", ""), out
    assert env["keys"].read_text(encoding="utf-8") == "ssh-ed25519 AAAA original\n"


def _enable_notes(env) -> Path:
    import json

    notes = env["home"] / "Notes"
    (env["persona"] / "persona_config.json").write_text(
        json.dumps({"notes_enabled": True, "notes_folder": str(notes)}), encoding="utf-8"
    )
    return notes


def test_append_to_a_hard_link_inside_the_notes_folder_is_not_auto_committed(env):
    # the notes folder skips the consent card, so this is the path where the link wrote straight through
    notes = _enable_notes(env)
    link = _link(env["keys"], notes / "ak")
    out = propose_write(path=str(link), content="ssh-ed25519 AAAA planted", op="append", persona_dir=env["persona"])
    assert "hard link" in out.get("error", "") and out.get("status") != "written", out
    assert env["keys"].read_text(encoding="utf-8") == "ssh-ed25519 AAAA original\n"


def test_a_link_planted_after_the_card_was_proposed_is_refused_at_commit(env):
    # commit re-runs the guard (TOCTOU): the card was for an ordinary file, which is then swapped for a link
    from brain.files import pending
    from brain.files.commit import commit_write
    from brain.memory.store import MemoryStore

    target = env["home"] / "Notes" / "log.md"
    target.write_text("start\n", encoding="utf-8")
    out = propose_write(path=str(target), content="ssh-ed25519 AAAA planted", op="append", persona_dir=env["persona"])
    rid = out["id"]
    target.unlink()
    _link(env["keys"], target)
    store = MemoryStore(env["persona"] / "memories.db")
    try:
        done = commit_write(env["persona"], rid, store=store)
    finally:
        store.close()
    assert done["ok"] is False and "hard link" in done["error"], done
    assert pending.get(env["persona"], rid)["status"] == "refused"
    assert env["keys"].read_text(encoding="utf-8") == "ssh-ed25519 AAAA original\n"


@pytest.mark.no_base_failure(reason="regression guard: an ordinary single-link append already works on the base; pins it against the #694 refusal")
def test_an_ordinary_append_inside_the_notes_folder_still_auto_commits(env):
    notes = _enable_notes(env)
    target = notes / "log.md"
    target.write_text("start\n", encoding="utf-8")
    out = propose_write(path=str(target), content="more", op="append", persona_dir=env["persona"])
    assert out.get("status") == "written", out
    assert target.read_text(encoding="utf-8").endswith("more")


def _swap_at_claim(monkeypatch, swap):
    """Run `swap()` when commit claims the card: after its guard re-run, before it opens the target."""
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


def test_a_hard_link_swapped_in_after_the_commit_guard_is_not_written(env, monkeypatch):
    # the guard's nlink check reads the path; commit then opened the path again, so a swap in between won
    target = env["home"] / "Notes" / "log.md"
    target.write_text("start\n", encoding="utf-8")
    rid = propose_write(path=str(target), content="ssh-ed25519 AAAA planted", op="append", persona_dir=env["persona"])["id"]

    def swap():
        target.unlink()
        _link(env["keys"], target)

    _swap_at_claim(monkeypatch, swap)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert env["keys"].read_text(encoding="utf-8") == "ssh-ed25519 AAAA original\n"
    assert pending.get(env["persona"], rid)["status"] == "refused"


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_a_link_planted_at_a_create_target_after_the_commit_guard_is_not_overwritten(env, monkeypatch, kind, request):
    # create used open("w"): a link planted after the guard was followed and the protected file truncated
    if kind == "symlink":
        request.getfixturevalue("requires_symlinks")
    target = env["home"] / "Notes" / "new.md"
    rid = propose_write(path=str(target), content="ssh-ed25519 AAAA planted", op="create", persona_dir=env["persona"])["id"]

    def plant():
        if kind == "symlink":
            target.symlink_to(env["keys"])
        else:
            _link(env["keys"], target)

    _swap_at_claim(monkeypatch, plant)
    done = _commit(env, rid)
    assert done["ok"] is False, done
    assert env["keys"].read_text(encoding="utf-8") == "ssh-ed25519 AAAA original\n"
    assert pending.get(env["persona"], rid)["status"] == "refused"
