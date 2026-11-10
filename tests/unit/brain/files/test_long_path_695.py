"""#695: a path the OS can't stat (too long) is refused, not raised.

``ENAMETOOLONG`` came out of ``resolved.exists()`` in the write guard and
``p.is_dir()`` in the read guard, after ``resolve()`` had already succeeded, so
the tools raised instead of answering ``{"error": ...}``. Driven through the
tools with a temp HOME and KINDLED_HOME; no mocked state files.
"""
import sys
from pathlib import Path

import pytest

from brain.tools.impls.propose_write import propose_write

# Windows takes these inputs as ordinary paths (long-path support is on; UTF-16 names may hold a lone
# surrogate; resolve() returns a looping link as is), so nothing raises and there is nothing to refuse there.
# test_windows_answers_every_one_of_these_without_raising checks that instead.
_POSIX = pytest.mark.skipif(sys.platform == "win32", reason="Windows accepts these paths; see the Windows test")


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


@_POSIX
@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("op", ["create", "append"])
def test_propose_write_refuses_a_too_long_path(env, which, op):
    out = propose_write(path=_long_paths(env["home"])[which], content="hi", op=op, persona_dir=env["persona"])
    assert "bad path" in out.get("error", ""), out


@_POSIX
@pytest.mark.parametrize("which", [0, 1])
def test_check_size_refuses_a_too_long_path(env, which):
    # commit calls check_size after check_write_target; it must not raise if the path stopped being usable
    from brain.files.write_guard import check_size

    r = check_size("hi", op="append", resolved=Path(_long_paths(env["home"])[which]))
    assert not r.ok and "bad path" in r.error


@_POSIX
@pytest.mark.parametrize("which", [0, 1])
def test_read_file_refuses_a_too_long_path(env, which):
    from brain.tools.impls.read_file import read_file

    out = read_file(path=_long_paths(env["home"])[which], persona_dir=env["persona"])
    assert "bad path" in out.get("error", "") and "content" not in out, out


@_POSIX
@pytest.mark.parametrize("which", [0, 1])
def test_list_directory_refuses_a_too_long_path(env, which):
    from brain.tools.impls.list_directory import list_directory

    out = list_directory(path=_long_paths(env["home"])[which], persona_dir=env["persona"])
    assert "bad path" in out.get("error", "") and "entries" not in out, out


@_POSIX
@pytest.mark.parametrize("which", [0, 1])
def test_read_guard_fails_closed_on_a_too_long_path(env, which):
    # the read tools refuse first, but the guard answers for any caller: unusable means denied, never a raise
    from brain.files.read_guard import is_read_denied

    assert is_read_denied(Path(_long_paths(env["home"])[which]), persona_dir=env["persona"]) is True


@_POSIX
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


@_POSIX
def test_propose_write_refuses_a_symlink_loop(env, requires_symlinks):
    # Python 3.12's Path.resolve() raises RuntimeError on a loop (3.13 returns a path); neither may escape
    a, b = env["home"] / "loopA", env["home"] / "loopB"
    a.symlink_to(b)
    b.symlink_to(a)
    out = propose_write(path=str(a / "x.md"), content="hi", op="create", persona_dir=env["persona"])
    assert "bad path" in out.get("error", ""), out


def test_propose_write_refuses_a_path_with_a_lone_surrogate(env):
    bad = "\udc80"  # the OS takes it (surrogateescape); "\ud800" was already refused before #695
    # a JSON tool argument can carry one; it isn't valid text, and later steps encode the path as UTF-8
    out = propose_write(path=str(env["home"] / f"a{bad}.md"), content="hi", op="create", persona_dir=env["persona"])
    assert "bad path" in out.get("error", ""), out


@pytest.mark.parametrize("bad", ["a\x00b", pytest.param("a\ud800b", marks=_POSIX)])
def test_read_guard_answers_denied_for_a_path_that_is_not_a_path(env, bad):
    # NUL or a lone surrogate: the read tools' resolve() refuses these first, but the guard itself must answer
    from brain.files.read_guard import is_read_denied

    assert is_read_denied(env["home"] / bad, persona_dir=env["persona"]) is True


@pytest.mark.no_base_failure(reason="pins one behaviour across Pythons: 3.12 already left a looping entry out (passes on the base there); 3.13 listed it as a file")
def test_listing_a_folder_with_a_looping_symlink_still_works(env, requires_symlinks):
    # an entry nothing can open is left out, as Python 3.12 always did; 3.13 used to list it as a "file"
    from brain.tools.impls.list_directory import list_directory

    d = env["home"] / "Docs"
    d.mkdir()
    (d / "note.md").write_text("x", encoding="utf-8")
    (d / "loopA").symlink_to(d / "loopB")
    (d / "loopB").symlink_to(d / "loopA")
    out = list_directory(path=str(d), persona_dir=env["persona"])
    assert [e["name"] for e in out["entries"]] == ["note.md"], out


@pytest.mark.no_base_failure(reason="Windows-only, and the base-code check runs on Linux; on Windows these were never raised, so this pins the contract: answer, don't raise")
@pytest.mark.skipif(sys.platform != "win32", reason="the Windows half of the contract; POSIX refuses these (tests above)")
def test_windows_answers_every_one_of_these_without_raising(env):
    from brain.files.read_guard import is_read_denied
    from brain.files.write_guard import check_size
    from brain.tools.impls.list_directory import list_directory
    from brain.tools.impls.read_file import read_file

    for raw in [*_long_paths(env["home"]), str(env["home"] / "a\ud800b")]:
        assert isinstance(propose_write(path=raw, content="hi", op="create", persona_dir=env["persona"]), dict)
        assert isinstance(read_file(path=raw, persona_dir=env["persona"]), dict)
        assert isinstance(list_directory(path=raw, persona_dir=env["persona"]), dict)
        assert isinstance(is_read_denied(Path(raw), persona_dir=env["persona"]), bool)
        assert check_size("hi", op="append", resolved=Path(raw)).ok in (True, False)
