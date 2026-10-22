"""The shutdown flag reaches every background model-call gate and every `claude` spawn site (C4, C7, C50, D2, D23)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from brain.bridge import cli_throttle

_BACKGROUND_SITES = {"generate", "probe"}
_CHAT_SITES = {"chat_text", "chat_stream_tools", "chat_stream_no_tools", "chat_mcp_tools"}


@pytest.fixture(autouse=True)
def _clean_throttle():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


def _idle_now():
    # Make the lull elapsed: the last message was long ago.
    cli_throttle._last_message_mono = float("-inf")


def test_flag_clear_keeps_the_gates_as_before():
    _idle_now()
    assert cli_throttle.shutdown_requested() is False
    assert cli_throttle.is_chat_idle() is True
    assert cli_throttle.slot_available() is True


def test_flag_set_closes_every_background_gate():
    _idle_now()
    cli_throttle.request_shutdown()
    assert cli_throttle.shutdown_requested() is True
    assert cli_throttle.is_chat_idle() is False
    assert cli_throttle.should_yield() is True
    assert cli_throttle.slot_available() is False
    assert cli_throttle.acquire_background() is False
    with cli_throttle.background_slot() as got:
        assert got is False


def test_reset_clears_the_flag():
    cli_throttle.request_shutdown()
    cli_throttle.reset()
    assert cli_throttle.shutdown_requested() is False


def test_shutdown_deferred_is_a_throttle_deferral():
    assert issubclass(cli_throttle.ShutdownDeferred, cli_throttle.ThrottleDeferred)


def test_every_lean_invocation_site_is_classified():
    """C7 guard: every `claude -p` spawn site in the lean-invocation inventory is either refused on shutdown
    (background) or chat-only; a new site fails here until it is classified."""
    from tests.unit.brain.bridge.test_provider_lean_invocation import _SITES

    assert set(_SITES) == _BACKGROUND_SITES | _CHAT_SITES
    assert not (_BACKGROUND_SITES & _CHAT_SITES)


def test_chat_sites_are_reached_only_from_a_chat_turn():
    """The chat entry points are not refused (a running chat turn finishes, D21), so their callers must be the chat
    turn only: chat/tool_loop.py, the server's _StreamingProxy, and Ollama's own generate -> chat delegation (refused
    at generate)."""
    import ast

    root = Path(__file__).resolve().parents[4] / "brain"
    callers: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("chat", "chat_stream")
            ):
                callers.setdefault(path.relative_to(root.parent).as_posix(), []).append(ast.unparse(node.func.value))
    assert set(callers) <= {"brain/chat/tool_loop.py", "brain/bridge/server.py", "brain/bridge/provider.py"}, callers
    assert callers.get("brain/bridge/provider.py", []) == ["self"]  # only Ollama's generate -> self.chat delegation


def test_background_sites_refuse_without_spawning():
    from brain.bridge.provider import ClaudeCliProvider, OllamaProvider

    cli_throttle.request_shutdown()
    with mock.patch.object(subprocess, "run") as run, mock.patch.object(subprocess, "Popen") as popen:
        with pytest.raises(cli_throttle.ShutdownDeferred):
            ClaudeCliProvider().generate("hi")
        with pytest.raises(cli_throttle.ShutdownDeferred):
            ClaudeCliProvider().probe("hi")
        assert not run.called and not popen.called
    with mock.patch("httpx.post") as post, mock.patch("httpx.stream") as stream:
        with pytest.raises(cli_throttle.ShutdownDeferred):
            OllamaProvider().generate("hi")
        assert not post.called and not stream.called


def test_chat_is_not_refused_on_shutdown():
    from brain.bridge.provider import ChatMessage, ClaudeCliProvider

    cli_throttle.request_shutdown()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with mock.patch.object(subprocess, "run", return_value=fake) as run:
        try:
            ClaudeCliProvider().chat([ChatMessage(role="user", content="hi")])
        except Exception:  # noqa: BLE001 - the empty fake output may not parse; only the spawn matters here
            pass
        assert run.called


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX session isolation")
@pytest.mark.parametrize("site", ["generate", "chat"])
def test_claude_children_start_in_their_own_session(site):
    """D23 (C59 unit half): a terminal Ctrl-C reaches the bridge only, not its in-flight claude calls."""
    from brain.bridge.provider import ChatMessage, ClaudeCliProvider

    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout='{"result": "x"}', stderr="")
    with mock.patch.object(subprocess, "run", return_value=fake) as run:
        try:
            if site == "generate":
                ClaudeCliProvider().generate("hi")
            else:
                ClaudeCliProvider().chat([ChatMessage(role="user", content="hi")])
        except Exception:  # noqa: BLE001
            pass
        assert run.called
        assert run.call_args.kwargs.get("start_new_session") is True


def test_login_expired_reads_the_auth_status():
    from brain.bridge import provider_auth

    provider_auth.reset()
    assert provider_auth.login_expired() is False
    provider_auth.note_cli_failure("Invalid API key · Please run /login")
    provider_auth.note_cli_failure("Invalid API key · Please run /login")
    assert provider_auth.login_expired() is (provider_auth.state()["status"] == "expired")
    provider_auth.reset()
