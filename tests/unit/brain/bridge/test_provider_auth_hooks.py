"""#246 — the ClaudeCliProvider feeds provider_auth at every auth-bearing site and gates generate()."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge import provider_auth
from brain.bridge.chat import StreamDone, StreamError
from brain.bridge.provider import ChatMessage, ClaudeCliProvider, ProviderError

LIVE = "Failed to authenticate: OAuth session expired and could not be refreshed"
MSGS = [ChatMessage(role="user", content="hi")]


@pytest.fixture(autouse=True)
def _reset():
    provider_auth.reset()
    yield
    provider_auth.reset()


def _run_result(rc: int, stdout: str, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = rc
    m.stdout = stdout
    m.stderr = stderr
    return m


def _auth_frame() -> str:
    return json.dumps({"is_error": True, "result": LIVE})


def _ok_frame(text: str = "hello") -> str:
    return json.dumps({"is_error": False, "result": text})


class _FakePopen:
    """Enough of subprocess.Popen for chat_stream: stdin, iterable stdout, wait, stderr."""

    def __init__(self, lines: list[str], rc: int = 0, stderr: str = ""):
        self.stdin = io.StringIO()
        self.stdout = iter(lines)
        self.stderr = io.StringIO(stderr)
        self._rc = rc
        self.pid = 4242

    def wait(self, timeout=None):
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass

    def kill(self):
        pass


def _stream_result_line(is_error: bool, text: str) -> str:
    return json.dumps({"type": "result", "is_error": is_error, "result": text}) + "\n"


# ---------------- C2: failure sites -> expired ----------------

def test_generate_exit_nonzero_marks_expired():
    with patch("subprocess.run", return_value=_run_result(1, _auth_frame())):
        with pytest.raises(RuntimeError):
            ClaudeCliProvider(model="haiku").generate("x")
    assert provider_auth.state()["status"] == "expired"


def test_generate_is_error_frame_marks_expired():
    with patch("subprocess.run", return_value=_run_result(0, _auth_frame())):
        with pytest.raises(RuntimeError):
            ClaudeCliProvider(model="haiku").generate("x")
    assert provider_auth.state()["status"] == "expired"


def test_chat_exit_nonzero_marks_expired():
    with patch("subprocess.run", return_value=_run_result(1, _auth_frame())):
        with pytest.raises(ProviderError):
            ClaudeCliProvider(model="sonnet").chat(MSGS)
    assert provider_auth.state()["status"] == "expired"


def test_chat_is_error_frame_marks_expired():
    with patch("subprocess.run", return_value=_run_result(0, _auth_frame())):
        with pytest.raises(ProviderError):
            ClaudeCliProvider(model="sonnet").chat(MSGS)
    assert provider_auth.state()["status"] == "expired"


def test_chat_stream_is_error_frame_marks_expired():
    fake = _FakePopen([_stream_result_line(True, LIVE)], rc=0)
    with patch("subprocess.Popen", return_value=fake):
        events = list(ClaudeCliProvider(model="sonnet").chat_stream(MSGS))
    assert any(isinstance(e, StreamError) for e in events)
    assert provider_auth.state()["status"] == "expired"


def test_chat_stream_exit_nonzero_stderr_only_marks_expired():
    fake = _FakePopen([], rc=1, stderr=LIVE)
    with patch("subprocess.Popen", return_value=fake):
        events = list(ClaudeCliProvider(model="sonnet").chat_stream(MSGS))
    assert any(isinstance(e, StreamError) for e in events)
    assert provider_auth.state()["status"] == "expired"


# ---------------- C3: success clears on each path ----------------

def test_generate_success_clears_expired(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(provider_auth, "_monotonic", lambda: t["now"])
    provider_auth.note_cli_failure(LIVE)
    t["now"] += provider_auth.REPROBE_S + 1  # the probe is granted
    with patch("subprocess.run", return_value=_run_result(0, _ok_frame())):
        assert ClaudeCliProvider(model="haiku").generate("x") == "hello"
    assert provider_auth.state()["status"] == "ok"


def test_chat_success_clears_expired():
    provider_auth.note_cli_failure(LIVE)
    with patch("subprocess.run", return_value=_run_result(0, _ok_frame())):
        ClaudeCliProvider(model="sonnet").chat(MSGS)
    assert provider_auth.state()["status"] == "ok"


def test_chat_stream_success_clears_expired():
    provider_auth.note_cli_failure(LIVE)
    fake = _FakePopen([_stream_result_line(False, "hello")], rc=0)
    with patch("subprocess.Popen", return_value=fake):
        events = list(ClaudeCliProvider(model="sonnet").chat_stream(MSGS))
    assert any(isinstance(e, StreamDone) for e in events)
    assert provider_auth.state()["status"] == "ok"


# ---------------- C4: generate() gate ----------------

def test_generate_gate_one_probe_per_interval(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(provider_auth, "_monotonic", lambda: t["now"])
    provider_auth.note_cli_failure(LIVE)
    t["now"] += provider_auth.REPROBE_S + 1
    run = MagicMock(return_value=_run_result(1, _auth_frame()))
    with patch("subprocess.run", run):
        p = ClaudeCliProvider(model="haiku")
        deferred = 0
        for _ in range(10):
            try:
                p.generate("x")
            except provider_auth.ProviderAuthDeferred:
                deferred += 1
            except RuntimeError:
                pass
    assert run.call_count == 1
    assert deferred == 9


def test_generate_gate_next_call_after_transition_is_deferred(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(provider_auth, "_monotonic", lambda: t["now"])
    provider_auth.note_cli_failure(LIVE)
    run = MagicMock()
    with patch("subprocess.run", run), pytest.raises(provider_auth.ProviderAuthDeferred):
        ClaudeCliProvider(model="haiku").generate("x")
    assert run.call_count == 0


def test_chat_and_chat_stream_are_never_gated(monkeypatch):
    provider_auth.note_cli_failure(LIVE)
    with patch("subprocess.run", return_value=_run_result(0, _ok_frame())):
        ClaudeCliProvider(model="sonnet").chat(MSGS)  # no ProviderAuthDeferred
    provider_auth.note_cli_failure(LIVE)
    fake = _FakePopen([_stream_result_line(False, "hello")], rc=0)
    with patch("subprocess.Popen", return_value=fake):
        list(ClaudeCliProvider(model="sonnet").chat_stream(MSGS))
