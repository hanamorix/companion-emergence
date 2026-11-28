"""S71: `ClaudeCliSession`, one live `claude` process that judges one prompt at a time and forgets each before the next.

The fake CLI (`tests/fake_claude_session.py`) is a real script run as a real child process. It keeps a conversation history like
the real CLI and logs what each process held when it answered, so these tests assert on what the PROCESS saw, not on what
the product code believes it sent. The hostile switches (a swallowed clear, a late clear, a duplicate result frame, death
between items, an error frame, a reply that never comes) are the ways the product code could end up asking an item on a
process that still holds an earlier one.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import brain.bridge.provider as provider_mod
from brain.bridge import cli_throttle, shutdown_progress
from brain.bridge.provider import (
    ClaudeCliProvider,
    ClaudeCliSession,
    _PerCallSession,
    judge_session,
)
from tests.fake_claude_session import install

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a #! script")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _session(tmp_path, monkeypatch, *, mode: str = "echo", die_after: int = 0, timeout: int = 10, system: str = "SYS TEXT"):
    fake = install(tmp_path, monkeypatch, mode=mode, die_after=die_after)
    return fake, ClaudeCliProvider(timeout_seconds=timeout).open_session(system)


def _held(reply: str) -> list[str]:
    return json.loads(reply)["history"]


def test_every_prompt_is_answered_by_a_process_that_holds_only_that_prompt(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    with session:
        replies = [session.generate(f"item {n}") for n in (1, 2, 3)]
    assert [_held(r) for r in replies] == [["item 1"], ["item 2"], ["item 3"]]
    # What the process itself held when it answered (not what the product code thinks it sent):
    assert [a["held"] for a in fake.answers()] == [["item 1"], ["item 2"], ["item 3"]]
    assert len(fake.events("spawn")) == 1
    # One clear before each later item, none before the first, each before its item's message.
    kinds = [e["event"] for e in fake.events() if e["event"] in {"spawn", "clear", "message"}]
    assert kinds == ["spawn", "message", "clear", "message", "clear", "message"]


def test_the_system_prompt_is_kept_and_its_file_is_removed_when_the_session_closes(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, system="THE INSTRUCTIONS")
    session.generate("a")
    session.generate("b")
    spawn = fake.events("spawn")[0]
    assert spawn["system"] == "THE INSTRUCTIONS"
    assert Path(spawn["system_path"]).exists()  # kept for the life of the process, across the clear
    session.close()
    assert not Path(spawn["system_path"]).exists()


def test_close_ends_the_process_and_is_safe_twice_and_when_never_opened(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    session.generate("a")
    pid = fake.events("spawn")[0]["pid"]
    before = threading.active_count()
    session.close()
    session.close()
    assert [e["pid"] for e in fake.events("exit")] == [pid] and not _pid_alive(pid)
    assert threading.active_count() <= before - 2  # the stdout reader and the stderr drain are joined
    _fake2, never_opened = _session(tmp_path, monkeypatch)
    never_opened.close()


def test_a_clear_the_cli_swallows_is_not_trusted_the_next_prompt_goes_to_a_fresh_process(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, mode="echo,swallow_clear")
    with session:
        replies = [session.generate(f"item {n}") for n in (1, 2, 3)]
    assert [_held(r) for r in replies] == [["item 1"], ["item 2"], ["item 3"]]
    assert len(fake.events("spawn")) == 3  # the old process held item 1: it never saw item 2
    assert [a["held"] for a in fake.answers()] == [["item 1"], ["item 2"], ["item 3"]]
    assert not any(_pid_alive(s["pid"]) for s in fake.events("spawn"))  # and none was left running


def test_a_clear_whose_confirmation_comes_late_is_still_waited_for(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, mode="echo,late_clear")
    with session:
        replies = [session.generate(f"item {n}") for n in (1, 2)]
    assert [_held(r) for r in replies] == [["item 1"], ["item 2"]]
    assert len(fake.events("spawn")) == 1
    assert [a["held"] for a in fake.answers()] == [["item 1"], ["item 2"]]


def test_an_extra_result_frame_for_one_message_never_becomes_the_next_prompts_reply(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, mode="echo,dup_result")
    with session:
        replies = [session.generate(f"item {n}") for n in (1, 2, 3)]
    assert [_held(r) for r in replies] == [["item 1"], ["item 2"], ["item 3"]]
    assert [a["held"] for a in fake.answers()] == [["item 1"], ["item 2"], ["item 3"]]
    assert len(fake.events("spawn")) >= 2  # the process that sent the stray frame was not reused


def test_a_process_that_died_between_items_is_replaced_not_asked(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, die_after=1)
    with session:
        first = session.generate("item 1")
        second = session.generate("item 2")
    assert _held(first) == ["item 1"] and _held(second) == ["item 2"]
    assert len(fake.events("spawn")) == 2 and len(fake.events("died")) >= 1


def test_a_failed_prompt_raises_what_generate_raises_and_the_next_prompt_gets_a_fresh_process(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    with session:
        session.generate("item 1")
        with pytest.raises(RuntimeError, match="boom"):
            session.generate("item 2 FAIL")
        pid_failed = fake.events("spawn")[0]["pid"]
        assert not _pid_alive(pid_failed)
        third = session.generate("item 3")
    assert _held(third) == ["item 3"]
    assert len(fake.events("spawn")) == 2


def test_a_reply_that_never_comes_times_out_on_one_deadline_and_the_process_is_discarded(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, timeout=1)
    with session:
        started = time.monotonic()
        # The fake keeps sending non-result frames: a per-frame wait would never fire.
        with pytest.raises(TimeoutError):
            session.generate("HANG")
        assert time.monotonic() - started < 6
        assert not _pid_alive(fake.events("spawn")[0]["pid"])
        assert _held(session.generate("after")) == ["after"]


def test_a_spawn_failure_raises_the_os_error_generate_raises_and_leaves_no_file_behind(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, system="WILL NOT RUN")
    paths: list[str] = []
    real = provider_mod._system_prompt_tempfile

    @contextlib.contextmanager
    def _recording(text):
        with real(text) as path:
            paths.append(path)
            yield path

    monkeypatch.setattr(provider_mod, "_system_prompt_tempfile", _recording)

    def _no_spawn(*a, **k):
        raise FileNotFoundError("no claude")

    monkeypatch.setattr(provider_mod.subprocess, "Popen", _no_spawn)
    with pytest.raises(FileNotFoundError):
        session.generate("x")
    assert paths and not Path(paths[0]).exists()


def test_the_process_starts_in_its_own_session_with_the_same_spawn_as_generate(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    real_popen = provider_mod.subprocess.Popen
    seen: list[dict] = []

    def _recording(cmd, **kwargs):
        seen.append({"cmd": list(cmd), **kwargs})
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(provider_mod.subprocess, "Popen", _recording)
    with session:
        session.generate("item 1")
    (spawn,) = seen
    assert spawn["start_new_session"] is True  # D23: a terminal Ctrl-C reaches the bridge only
    assert spawn["cwd"] == str(tmp_path) and "CAVEMAN_DEFAULT_MODE" in spawn["env"]
    cmd = spawn["cmd"]
    assert cmd[:3] == ["claude", "-p", "--dangerously-skip-permissions"]
    assert cmd[cmd.index("--model") + 1] == provider_mod.model_for_tier(provider_mod.TIER_INTERACTIVE_CHAT)
    assert "--strict-mcp-config" in cmd and "--tools=" in cmd and "--system-prompt-file" in cmd


def test_a_stopping_bridge_defers_before_any_process_is_started(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_throttle, "shutdown_requested", lambda: True)
    with pytest.raises(cli_throttle.ShutdownDeferred):
        session.generate("x")
    assert fake.events("spawn") == []


def test_a_reply_of_the_old_conversation_arriving_after_the_clear_is_never_taken_as_the_reply(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, mode="echo,stale_before_answer")
    with session:
        first = session.generate("item 1")
        with pytest.raises(RuntimeError, match="not the reply"):  # the stale frame came before the real answer
            session.generate("item 2")
        third = session.generate("item 3")
    assert _held(first) == ["item 1"] and _held(third) == ["item 3"]
    assert not any(_pid_alive(s["pid"]) for s in fake.events("spawn"))  # the process that sent it was discarded
    assert len(fake.events("spawn")) == 2


def test_a_second_clear_style_result_before_the_answer_is_never_taken_as_the_reply(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, mode="echo,zero_turn_before_answer")
    with session:
        session.generate("item 1")
        with pytest.raises(RuntimeError, match="not the reply"):
            session.generate("item 2")
        third = session.generate("item 3")
    assert _held(third) == ["item 3"] and len(fake.events("spawn")) == 2


def test_a_clear_answered_with_an_error_frame_is_not_trusted(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch, mode="echo,clear_error")
    with session:
        replies = [session.generate(f"item {n}") for n in (1, 2)]
    assert [_held(r) for r in replies] == [["item 1"], ["item 2"]]
    assert len(fake.events("spawn")) == 2


def test_an_expired_login_defers_before_any_process_is_started(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(provider_mod.provider_auth, "should_skip_background", lambda: True)
    with pytest.raises(provider_mod.provider_auth.ProviderAuthDeferred):
        session.generate("x")
    assert fake.events("spawn") == []


def test_the_hard_exit_gate_is_checked_before_a_send_after_a_wait_and_after_the_call(tmp_path, monkeypatch):
    fake, session = _session(tmp_path, monkeypatch)
    calls: list[int] = []
    monkeypatch.setattr(shutdown_progress, "abandon_gate", lambda: calls.append(1))
    with session:
        session.generate("item 1")
        first_call = len(calls)
        session.generate("item 2")
    # spawn, send, wait (at least one frame), end of call: the second call also sends a clear and waits for it.
    assert first_call >= 4 and len(calls) - first_call >= 7


def test_judge_session_gives_a_live_session_only_to_the_claude_cli_provider(tmp_path, monkeypatch):
    assert isinstance(judge_session(ClaudeCliProvider(), "S"), ClaudeCliSession)

    class _Stub:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, *, system=None):
            self.calls.append((prompt, system))
            return "reply"

    stub = _Stub()
    session = judge_session(stub, "SYSTEM")
    assert isinstance(session, _PerCallSession)
    assert session.generate("p1") == "reply" and session.generate("p2") == "reply"
    session.close()
    assert stub.calls == [("p1", "SYSTEM"), ("p2", "SYSTEM")]
    # A mock has every attribute, including an open_session: it still gets one call per prompt.
    mock = MagicMock()
    assert isinstance(judge_session(mock, "S"), _PerCallSession)
