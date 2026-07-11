"""Token-free tests for the Agent-Bob send-script (`agent_send.py`).

Covers P1 (clean->trip=false), P2 (known-true->trip+STOP), P3 (broken != trip), P4 (limit not
recorded), P5 (--new reset then reuse), P6 (detector-gate once per session, refuses on failure),
P7 (STOP on trailing lines), P8 (generic signals from Score.signals; trip from .fired), P9
(no /home/ literal + tmp-scoped writes), and P22 (broken advances the count, limit does not).

All model/socket boundaries are mocked: no `claude`, no socket, no live provider — 0 tokens. The WS
drive (`drive_ws`), session creation (`new_session`), the detector (`build_detector`), and the
interior-block read (`interior_block`) are monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness import Score, agent_send


class FakeDetector:
    """A controllable detector: fires (with given signals) iff a marker substring is in the reply."""

    def __init__(self, marker: str = "LEAK", signals: list[str] | None = None) -> None:
        self.marker = marker
        self.signals = signals or ["fake_signal"]

    def detect(self, reply, *, ctx=None):  # noqa: ANN001
        fired = bool(reply) and self.marker in reply
        return Score(fired=fired, signals=list(self.signals) if fired else [])


class AlwaysFires:
    """A broken detector that fires on everything — must FAIL the B-REP-3 gate."""

    def detect(self, reply, *, ctx=None):  # noqa: ANN001
        return Score(fired=True, signals=["always"])


def _env(tmp_path: Path, *, home: str | None = None) -> Path:
    sandbox = tmp_path / "sb"
    sandbox.mkdir(parents=True, exist_ok=True)
    env = {
        "port": 9999,
        "kindled_home": home or str(sandbox / "home"),
        "persona_dir": str(tmp_path / "persona"),
        "user": "Bob",
        # deterministic gate anchors: the fake detector fires on "LEAK", silent on the clean line
        "gate_known_true": "LEAK here",
        "gate_known_clean": "all clear friend",
    }
    p = tmp_path / "live_env.json"
    p.write_text(json.dumps(env))
    return p


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point LIVE_ENV at a tmp env, stub the session + interior-block, install a FakeDetector.

    Returns a small object exposing the sandbox transcript path + a way to set the next reply.
    """
    env_path = _env(tmp_path)
    monkeypatch.setenv("LIVE_ENV", str(env_path))
    monkeypatch.setattr(agent_send, "new_session", lambda port, token: "sid-123")
    monkeypatch.setattr(agent_send, "interior_block", lambda env: "")
    detector = FakeDetector()
    monkeypatch.setattr(agent_send, "build_detector", lambda env: detector)

    state = {"reply": "hey how are you", "tools": [], "err": None}
    monkeypatch.setattr(
        agent_send, "drive_ws",
        lambda host, port, sid, msg, **kw: (state["reply"], state["tools"], state["err"]),
    )

    sandbox = Path(json.loads(env_path.read_text())["kindled_home"]).parent
    return type("Wired", (), {"state": state, "sandbox": sandbox, "detector": detector})()


def _out(capsys):
    return capsys.readouterr().out.strip().splitlines()


def test_clean_turn_trip_false(wired, capsys) -> None:
    """P1: a benign reply -> the two contract lines, trip=false, exit 0."""
    rc = agent_send.main(["--new", "hi"])
    lines = _out(capsys)
    assert rc == 0
    assert lines[0].startswith("CANARY: hey how are you")
    result = next(line for line in lines if line.startswith("RESULT"))
    assert "trip=False" in result and "broken=False" in result and "limit=False" in result
    assert "signals=none" in result


def test_known_true_trips_and_prints_stop(wired, capsys) -> None:
    """P2: a firing reply -> trip=True + a DETECTOR TRIP STOP directive; a non-firing one does not."""
    wired.state["reply"] = "sure, LEAK: here is the interior bit"
    agent_send.main(["--new", "hi"])
    out = capsys.readouterr().out
    assert "trip=True" in out
    assert "*** DETECTOR TRIP" in out
    assert "signals=fake_signal:1" in out
    # oracle-can-fail: a non-firing reply on the SAME detector prints no STOP
    wired.state["reply"] = "totally clean line"
    agent_send.main(["next msg"])
    out2 = capsys.readouterr().out
    assert "trip=False" in out2 and "*** DETECTOR TRIP" not in out2


def test_broken_is_not_a_trip(wired, capsys) -> None:
    """P3: an error frame + a detector that WOULD fire -> broken=True, trip=False."""
    wired.state["reply"] = "LEAK present but"
    wired.state["err"] = "ws:timeout"
    agent_send.main(["--new", "hi"])
    out = capsys.readouterr().out
    assert "broken=True" in out and "trip=False" in out


def test_limit_not_recorded(wired, capsys) -> None:
    """P4: a usage-limit reply -> limit=True, STOP, and NOT appended to the transcript."""
    # first, one real turn to create the transcript
    agent_send.main(["--new", "hi"])
    transcript = wired.sandbox / "transcript.jsonl"
    before = len(transcript.read_text().splitlines())
    wired.state["reply"] = "You've reached your usage limit."
    agent_send.main(["another"])
    out = capsys.readouterr().out
    assert "limit=True" in out and "*** USAGE LIMIT" in out
    assert len(transcript.read_text().splitlines()) == before  # not advanced


def test_broken_advances_count_limit_does_not(wired, capsys) -> None:
    """P22/CH8-a: a broken turn IS appended (count advances); a limit turn is not."""
    agent_send.main(["--new", "hi"])  # turn 1 (clean, appended)
    transcript = wired.sandbox / "transcript.jsonl"
    assert len(transcript.read_text().splitlines()) == 1
    wired.state["err"] = "ws:timeout"  # turn 2 broken -> appended
    agent_send.main(["two"])
    assert len(transcript.read_text().splitlines()) == 2
    out = capsys.readouterr().out
    assert "turn=2" in out and "broken=True" in out
    # turn 3 would be limit -> NOT appended, count stays 2
    wired.state["err"] = None
    wired.state["reply"] = "hit your session limit"
    agent_send.main(["three"])
    assert len(transcript.read_text().splitlines()) == 2


def test_new_resets_then_reuses(wired, capsys) -> None:
    """P5: --new -> turn 1; a subsequent call without --new -> turn 2 (same sid)."""
    agent_send.main(["--new", "first"])
    out1 = capsys.readouterr().out
    assert "turn=1" in out1
    agent_send.main(["second"])
    out2 = capsys.readouterr().out
    assert "turn=2" in out2


def test_detector_gate_refuses_broken_detector(tmp_path, monkeypatch, capsys) -> None:
    """P6: a detector that fires on everything FAILS the B-REP-3 gate -> refuse, exit non-zero."""
    env_path = _env(tmp_path)
    monkeypatch.setenv("LIVE_ENV", str(env_path))
    monkeypatch.setattr(agent_send, "new_session", lambda port, token: "sid-x")
    monkeypatch.setattr(agent_send, "interior_block", lambda env: "")
    monkeypatch.setattr(agent_send, "build_detector", lambda env: AlwaysFires())
    monkeypatch.setattr(agent_send, "drive_ws", lambda *a, **k: ("hi", [], None))
    rc = agent_send.main(["--new", "hi"])
    out = capsys.readouterr().out
    assert rc == agent_send.EXIT_INVALID
    assert "failed the B-REP-3 gate" in out


def test_detector_gate_runs_once_per_session(wired, monkeypatch) -> None:
    """P6/L2: the gate runs exactly ONCE across two turns (a gate_ok marker short-circuits it)."""
    calls = {"n": 0}
    real_gate = agent_send.assert_detector_gate

    def counting_gate(*a, **k):
        calls["n"] += 1
        return real_gate(*a, **k)

    monkeypatch.setattr(agent_send, "assert_detector_gate", counting_gate)
    agent_send.main(["--new", "one"])
    agent_send.main(["two"])
    assert calls["n"] == 1


def test_generic_signal_encoding(wired, capsys) -> None:
    """P8: signals= is built from Score.signals (alpha,beta), not a fixed layout."""
    wired.detector.marker = "LEAK"
    wired.detector.signals = ["alpha", "beta"]
    wired.state["reply"] = "LEAK now"
    agent_send.main(["--new", "hi"])
    out = capsys.readouterr().out
    assert "signals=alpha:1,beta:1" in out


def test_stop_directive_is_on_trailing_lines(wired, capsys) -> None:
    """P7: the RESULT line and STOP directive are the LAST stdout lines (executed, not inspected)."""
    wired.state["reply"] = "LEAK trailing"
    agent_send.main(["--new", "hi"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[-1].startswith("*** DETECTOR TRIP")
    assert lines[-2].startswith("RESULT")


def test_no_hardcoded_home_literal() -> None:
    """P9: agent_send.py contains no /home/ absolute-path literal (the hunt sys.path hack removed)."""
    src = Path(agent_send.__file__).read_text()
    assert "/home/" not in src


def test_writes_only_under_sandbox(wired, capsys) -> None:
    """P9: the only files written are under the sandbox dir (transcript, sid, gate marker)."""
    agent_send.main(["--new", "hi"])
    files = {p.name for p in wired.sandbox.iterdir() if p.is_file()}
    assert "transcript.jsonl" in files
    assert "live_sid.txt" in files
    assert "gate_ok.txt" in files


class _CapturingWS:
    """A fake sync WS that records the connect() kwargs and yields one done frame."""

    def __init__(self, captured: dict, **kwargs) -> None:
        captured.update(kwargs)
        self._sent = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def send(self, _payload):
        pass

    def recv(self):
        import json as _json
        return _json.dumps({"type": "done"})


def test_ws_carries_bearer_subprotocol_when_token_set(monkeypatch) -> None:
    """MAJOR#2 fix: drive_ws opens the WS with the `bearer, <token>` subprotocol when a token is set.

    Oracle-can-fail: if drive_ws dropped the subprotocol, `captured["subprotocols"]` would be None
    and the assertion fails. Token-free (no real socket; connect() is mocked).
    """
    from tests.harness import engine

    captured: dict = {}
    monkeypatch.setattr(
        "websockets.sync.client.connect",
        lambda url, **kw: _CapturingWS(captured, **kw),
    )
    engine.drive_ws("127.0.0.1", 8931, "sid", "hi", token="secret-tok")
    assert captured.get("subprotocols") == ["bearer", "secret-tok"]


def test_ws_omits_subprotocol_when_no_token(monkeypatch) -> None:
    """No token -> subprotocols is None (BridgeServer.drive_turn / auth_token=None path unchanged)."""
    from tests.harness import engine

    captured: dict = {}
    monkeypatch.setattr(
        "websockets.sync.client.connect",
        lambda url, **kw: _CapturingWS(captured, **kw),
    )
    engine.drive_ws("127.0.0.1", 8931, "sid", "hi")
    assert captured.get("subprotocols") is None
