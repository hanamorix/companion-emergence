"""#122 — every claude spawn runs in a working directory with no memory-bearing ancestor."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from brain.bridge import provider as provider_mod


@pytest.fixture(autouse=True)
def _reset_warned():
    provider_mod._claude_work_dir_reset_for_tests()
    yield
    provider_mod._claude_work_dir_reset_for_tests()


def _clean_root(tmp_path: Path, name: str) -> Path:
    """A candidate root with a clean ancestor chain (tmp_path itself has no markers)."""
    d = tmp_path / name
    d.mkdir()
    return d


MARKERS = (".claude/CLAUDE.md", "CLAUDE.md", "CLAUDE.local.md", ".claude", ".git")


# C2 — predicate + idempotence
def test_returns_dir_with_no_memory_ancestor(tmp_path: Path, monkeypatch):
    t = _clean_root(tmp_path, "t")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(_clean_root(tmp_path, "home")))
    d = provider_mod._claude_work_dir()
    assert d is not None
    p = Path(d)
    assert p.is_dir()
    for anc in [p, *p.parents]:
        for m in MARKERS:
            assert not (anc / m).exists(), f"{anc/m} would leak"
    assert provider_mod._claude_work_dir() == d


@pytest.mark.parametrize("marker", MARKERS)
def test_skips_tempdir_candidate_with_memory_ancestor(tmp_path: Path, monkeypatch, marker):
    t = _clean_root(tmp_path, "t")
    (t / marker).parent.mkdir(parents=True, exist_ok=True)
    if marker.endswith((".claude", ".git")):
        (t / marker).mkdir(exist_ok=True)
    else:
        (t / marker).write_text("x")
    home = _clean_root(tmp_path, "home")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    assert provider_mod._claude_work_dir() == str((home / "claude-work").resolve())


# C3 — both candidates dirty → None + one WARNING
def test_no_clean_candidate_returns_none_and_warns_once(tmp_path: Path, monkeypatch, caplog):
    t = _clean_root(tmp_path, "t")
    (t / "CLAUDE.md").write_text("x")
    home = _clean_root(tmp_path, "home")
    (home / ".claude").mkdir()
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
        assert provider_mod._claude_work_dir() is None
        assert provider_mod._claude_work_dir() is None
    warns = [r for r in caplog.records if "working directory" in r.getMessage()]
    assert len(warns) == 1
    assert "#252" not in warns[0].getMessage()  # non-Windows: generic text


def test_windows_warning_names_the_issue(tmp_path: Path, monkeypatch, caplog):
    t = _clean_root(tmp_path, "t")
    (t / "CLAUDE.md").write_text("x")
    home = _clean_root(tmp_path, "home")
    (home / ".claude").mkdir()
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    monkeypatch.setattr(sys, "platform", "win32")
    with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
        assert provider_mod._claude_work_dir() is None
    assert any("#252" in r.getMessage() for r in caplog.records)


# C2 win32-clean arm — no os.getuid on Windows
def test_simulated_windows_clean_profile_returns_tempdir_candidate(tmp_path: Path, monkeypatch):
    t = _clean_root(tmp_path, "t")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(_clean_root(tmp_path, "home")))
    # os.name cannot be faked on macOS (pathlib would instantiate WindowsPath); the Windows hazard
    # is the missing os.getuid attribute, so remove exactly that and set the platform tag.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(os, "getuid", raising=False)
    assert provider_mod._claude_work_dir() == str((t / "companion-emergence-claude-work").resolve())


@pytest.mark.skipif(os.name != "posix", reason="uid ownership check is POSIX-only")
def test_foreign_owned_candidate_is_skipped(tmp_path: Path, monkeypatch):
    t = _clean_root(tmp_path, "t")
    (t / "companion-emergence-claude-work").mkdir()
    home = _clean_root(tmp_path, "home")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    foreign_r = (t / "companion-emergence-claude-work").resolve()  # resolved BEFORE patching stat
    real_stat = Path.stat

    def fake_stat(self, *a, **k):
        st = real_stat(self, *a, **k)
        if self == foreign_r:
            return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid + 1, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
        return st

    monkeypatch.setattr(Path, "stat", fake_stat)  # only the first candidate looks foreign-owned
    assert provider_mod._claude_work_dir() == str((home / "claude-work").resolve())


# C4 — mkdir failure on the first candidate falls through; on all → None
def test_mkdir_failure_falls_through_to_next_candidate(tmp_path: Path, monkeypatch):
    t = _clean_root(tmp_path, "t")
    home = _clean_root(tmp_path, "home")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    real_mkdir = Path.mkdir

    def flaky(self, *a, **k):
        if "companion-emergence-claude-work" in str(self):
            raise OSError("read-only")
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", flaky)
    assert provider_mod._claude_work_dir() == str((home / "claude-work").resolve())


def test_no_candidate_creatable_returns_none_and_warns_once(tmp_path: Path, monkeypatch, caplog):
    t = _clean_root(tmp_path, "t")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(_clean_root(tmp_path, "home")))
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(OSError("ro")))
    with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
        assert provider_mod._claude_work_dir() is None
        assert provider_mod._claude_work_dir() is None
    assert len([r for r in caplog.records if "working directory" in r.getMessage()]) == 1


def test_permission_error_on_first_candidate_does_not_abort_the_loop(tmp_path: Path, monkeypatch):
    t = _clean_root(tmp_path, "t")
    home = _clean_root(tmp_path, "home")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    real_exists = Path.exists

    def hardened(self):
        if str(self).startswith(str(t)):
            raise PermissionError(13, "hardened parent")
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", hardened)
    assert provider_mod._claude_work_dir() == str((home / "claude-work").resolve())


# ---------------- C1: every spawn site passes cwd= ----------------
import io  # noqa: E402
import json  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from brain.bridge.provider import ChatMessage, ClaudeCliProvider  # noqa: E402

MSGS = [ChatMessage(role="user", content="hi")]


def _ok_run() -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps({"is_error": False, "result": "hello"})
    m.stderr = ""
    return m


class _FakePopen:
    def __init__(self, lines, rc=0):
        self.stdin = io.StringIO()
        self.stdout = iter(lines)
        self.stderr = io.StringIO("")
        self._rc = rc
        self.pid = 1

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass

    def kill(self):
        pass


@pytest.fixture
def clean_workdir(tmp_path: Path, monkeypatch) -> str:
    t = _clean_root(tmp_path, "t")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(_clean_root(tmp_path, "home")))
    d = provider_mod._claude_work_dir()
    assert d is not None
    return d


def test_generate_spawns_in_neutral_cwd(clean_workdir):
    run = MagicMock(return_value=_ok_run())
    with patch("subprocess.run", run):
        ClaudeCliProvider(model="haiku").generate("x")
    assert run.call_args.kwargs["cwd"] == clean_workdir


def test_chat_spawns_in_neutral_cwd(clean_workdir):
    run = MagicMock(return_value=_ok_run())
    with patch("subprocess.run", run):
        ClaudeCliProvider(model="sonnet").chat(MSGS)
    assert run.call_args.kwargs["cwd"] == clean_workdir


def test_chat_stream_spawns_in_neutral_cwd(clean_workdir):
    fake = _FakePopen([json.dumps({"type": "result", "is_error": False, "result": "hello"}) + "\n"])
    popen = MagicMock(return_value=fake)
    with patch("subprocess.Popen", popen):
        list(ClaudeCliProvider(model="sonnet").chat_stream(MSGS))
    assert popen.call_args.kwargs["cwd"] == clean_workdir


def test_mcp_tools_path_spawns_in_neutral_cwd(clean_workdir, tmp_path: Path):
    run = MagicMock(return_value=_ok_run())
    tools = [{"name": "noop", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    with patch("subprocess.run", run):
        ClaudeCliProvider(model="sonnet").chat(MSGS, tools=tools, options={"persona_dir": str(tmp_path)})
    assert run.call_args.kwargs["cwd"] == clean_workdir


def test_spawn_without_neutral_dir_keeps_status_quo(tmp_path: Path, monkeypatch):
    t = _clean_root(tmp_path, "t")
    (t / "CLAUDE.md").write_text("x")
    home = _clean_root(tmp_path, "home")
    (home / ".claude").mkdir()
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    monkeypatch.setenv("KINDLED_HOME", str(home))
    run = MagicMock(return_value=_ok_run())
    with patch("subprocess.run", run):
        ClaudeCliProvider(model="haiku").generate("x")
    assert run.call_args.kwargs.get("cwd") is None
