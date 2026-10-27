"""#482: the streaming claude path drained only stdout and never reaped or killed the child it told to stop.

* stderr was a pipe nothing read while the process ran, so a CLI that wrote more than a pipe buffer to stderr blocked, stdout stopped,
  and the turn ended only when the idle watchdog fired.
* when stdout reached EOF with no result frame, `proc.wait()` had no timeout, so a child still running hung the generator.
* after `terminate()` (idle timeout, generator close) and on the result-frame path the child was never waited on, and one that ignored
  SIGTERM kept running.
The tests run REAL child processes (a stand-in for the CLI) through the real `chat_stream`. POSIX only: they rely on signals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

import brain.bridge.provider as provider
from brain.bridge.chat import StreamDone, StreamError, TextDelta
from brain.bridge.provider import ChatMessage, ClaudeCliProvider

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="relies on POSIX signals")

MSGS = [ChatMessage(role="user", content="hi")]
_STAND_IN_LIFETIME_S = 60  # how long the stand-in child that never exits would live if nothing killed it
DELTA = json.dumps(
    {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel"}}}
)
RESULT = json.dumps({"type": "result", "result": "hello", "is_error": False})


def _alive(proc: subprocess.Popen) -> bool:
    """True while the child this test spawned is running: the Popen object's own ``poll()`` (portable; there is no /proc on macOS,
    and a pid alone can be reused). The child has printed its first frame before this is asked, so it has started."""
    return proc.poll() is None


def _reaped(proc: subprocess.Popen) -> bool:
    """True when the child was waited for (its exit status is collected) and the pid no longer exists. The Popen object is the one the
    provider used, so ``returncode`` is set exactly when the provider reaped it; the pid probe alone could be fooled by a reused pid."""
    if proc.returncode is None:
        return False
    try:
        os.kill(proc.pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.fixture
def child(monkeypatch):
    """Run `chat_stream` against a stand-in child running ``script`` instead of the claude CLI."""
    spawned: list[subprocess.Popen] = []
    real_popen = subprocess.Popen
    box = {"script": ""}

    def fake_popen(cmd, **kwargs):
        proc = real_popen([sys.executable, "-c", box["script"]], **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(provider, "_child_exit_grace_seconds", lambda: 0.5, raising=False)
    # Far longer than any test needs: a stream that stalls ends with a StreamError at the watchdog, which the assertions below
    # catch by event type, so no test needs a wall-clock bound that a loaded host could exceed.
    monkeypatch.setattr(provider, "_stream_first_event_seconds", lambda: 30.0)
    monkeypatch.setattr(provider, "_stream_per_event_idle_seconds", lambda: 30.0)

    def run(script: str):
        box["script"] = script
        return ClaudeCliProvider(probe_model="sonnet").chat_stream(MSGS)

    yield run, spawned
    for proc in spawned:  # never leave a stand-in running
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_a_flood_on_stderr_does_not_stall_the_stream(child):
    run, spawned = child
    script = (
        "import sys\n"
        "sys.stderr.write('e' * 300000)\n"  # several pipe buffers; blocks here unless something drains stderr
        "sys.stderr.flush()\n"
        f"print({RESULT!r}, flush=True)\n"
    )
    events = list(run(script))
    # StreamDone, not the StreamError the idle watchdog would produce after 30 s: stderr was drained while the child ran.
    assert [type(e) for e in events] == [StreamDone]
    assert events[0].content == "hello"


def test_stderr_text_still_reaches_the_failure_detail(child):
    run, _ = child
    script = "import sys\nsys.stderr.write('boom: it broke\\n')\nsys.exit(3)\n"
    events = list(run(script))
    assert len(events) == 1 and isinstance(events[0], StreamError)
    assert events[0].stage == "claude_cli_exit"
    assert "exit 3" in events[0].detail and "boom: it broke" in events[0].detail


def test_stdout_eof_with_a_child_that_never_exits_does_not_hang_and_reaps_it(child):
    run, spawned = child
    script = f"import os, time\nos.close(1)\ntime.sleep({_STAND_IN_LIFETIME_S})\n"
    t0 = time.monotonic()
    events = list(run(script))
    # The provider gives up on the child after the 0.5 s grace and kills it. A wait with no timeout would last the child's whole
    # life: half of it is a bound no loaded host reaches and the regression always does.
    assert time.monotonic() - t0 < _STAND_IN_LIFETIME_S / 2
    assert len(events) == 1 and isinstance(events[0], StreamError)
    assert events[0].stage == "claude_cli_exit"
    assert _reaped(spawned[0])


def test_an_idle_timeout_kills_and_reaps_a_child_that_ignores_sigterm(child, monkeypatch):
    run, spawned = child
    monkeypatch.setattr(provider, "_stream_first_event_seconds", lambda: 0.5)
    script = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"
    events = list(run(script))
    assert len(events) == 1 and isinstance(events[0], StreamError)
    assert events[0].stage == "claude_cli_idle_timeout"
    assert _reaped(spawned[0])


def test_closing_the_generator_kills_and_reaps_the_child(child):
    run, spawned = child
    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"print({DELTA!r}, flush=True)\n"
        "time.sleep(60)\n"
    )
    gen = run(script)
    first = next(gen)
    assert isinstance(first, TextDelta)
    assert _alive(spawned[0])
    gen.close()
    assert _reaped(spawned[0])


def test_a_result_frame_does_not_leave_a_lingering_child_behind(child):
    run, spawned = child
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"print({RESULT!r}, flush=True)\n"
        "time.sleep(60)\n"
    )
    events = list(run(script))
    assert [type(e) for e in events] == [StreamDone]
    assert _reaped(spawned[0])


def test_a_child_that_exits_on_its_own_after_the_result_frame_is_just_reaped(child):
    run, spawned = child
    script = f"print({RESULT!r}, flush=True)\n"
    events = list(run(script))
    assert [type(e) for e in events] == [StreamDone]
    assert spawned[0].returncode == 0


def test_the_teardown_waits_are_one_named_value_not_literals():
    """Code review 3a: `child_exit_grace_seconds = 5.0` duplicated a hard-coded `reader.join(timeout=5)`. Every bound in the teardown
    (the wait for the child, the reader thread's join, the stderr drain's join) comes from the one tunable."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(provider))
    literal_timeouts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("join", "wait"):
            for kw in node.keywords:
                if kw.arg == "timeout" and isinstance(kw.value, ast.Constant):
                    literal_timeouts.append((node.func.attr, kw.value.value, node.lineno))
    assert literal_timeouts == [], literal_timeouts
    drain_text = inspect.signature(provider._StderrDrain.text).parameters["wait_s"]
    assert drain_text.default is None  # None means the grace value, not a second number
