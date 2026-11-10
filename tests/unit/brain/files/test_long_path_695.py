"""#695: a path the OS can't stat (too long) is refused, not raised.

``ENAMETOOLONG`` came out of ``resolved.exists()`` in the write guard and
``p.is_dir()`` in the read guard, after ``resolve()`` had already succeeded, so
the tools raised instead of answering ``{"error": ...}``. Driven through the
tools with a temp HOME and KINDLED_HOME; no mocked state files.
"""
from pathlib import Path

import pytest

from brain.tools.impls.propose_write import propose_write


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    kh = tmp_path / "kindled"
    persona = kh / "personas" / "canary"
    for d in (home, persona):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("KINDLED_HOME", str(kh))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: home)
    return {"home": home, "persona": persona}


def _long_paths(home: Path) -> list[str]:
    # one component over NAME_MAX (255), and a whole path over PATH_MAX (1024 macOS, 4096 Linux)
    return [str(home / ("a" * 300) / "x.md"), str(home) + "/b" * 2600]


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("op", ["create", "append"])
def test_propose_write_refuses_a_too_long_path(env, which, op):
    out = propose_write(path=_long_paths(env["home"])[which], content="hi", op=op, persona_dir=env["persona"])
    assert "bad path" in out.get("error", ""), out


@pytest.mark.parametrize("which", [0, 1])
def test_check_size_refuses_a_too_long_path(env, which):
    # commit calls check_size after check_write_target; it must not raise if the path stopped being usable
    from brain.files.write_guard import check_size

    r = check_size("hi", op="append", resolved=Path(_long_paths(env["home"])[which]))
    assert not r.ok and "bad path" in r.error


@pytest.mark.parametrize("which", [0, 1])
def test_read_file_refuses_a_too_long_path(env, which):
    from brain.tools.impls.read_file import read_file

    out = read_file(path=_long_paths(env["home"])[which], persona_dir=env["persona"])
    assert "bad path" in out.get("error", "") and "content" not in out, out


@pytest.mark.parametrize("which", [0, 1])
def test_list_directory_refuses_a_too_long_path(env, which):
    from brain.tools.impls.list_directory import list_directory

    out = list_directory(path=_long_paths(env["home"])[which], persona_dir=env["persona"])
    assert "bad path" in out.get("error", "") and "entries" not in out, out


@pytest.mark.parametrize("which", [0, 1])
def test_read_guard_fails_closed_on_a_too_long_path(env, which):
    # the read tools refuse first, but the guard answers for any caller: unusable means denied, never a raise
    from brain.files.read_guard import is_read_denied

    assert is_read_denied(Path(_long_paths(env["home"])[which]), persona_dir=env["persona"]) is True


def test_approving_a_card_whose_path_is_too_long_refuses_it(env):
    # commit re-runs the guard (TOCTOU); an unusable path must come back as a refusal and close the card
    from datetime import UTC, datetime

    from brain.files import pending
    from brain.files.commit import commit_write
    from brain.memory.store import MemoryStore

    rid = pending.create(env["persona"], op="create", resolved_path=_long_paths(env["home"])[0],
                         content="hi", now=datetime.now(UTC))
    store = MemoryStore(env["persona"] / "memories.db")
    try:
        out = commit_write(env["persona"], rid, store=store)
    finally:
        store.close()
    assert out == {"ok": False, "error": out["error"]} and "bad path" in out["error"], out
    assert pending.get(env["persona"], rid)["status"] == "refused"
