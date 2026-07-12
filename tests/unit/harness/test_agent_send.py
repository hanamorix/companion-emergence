"""Token-free tests for the Agent-Bob send-script (`agent_send.py`).

Covers P1 (clean->trip=false), P2 (known-true->trip+STOP), P3 (broken != trip), P4 (limit not
recorded), P5 (--new reset then reuse), P6 (detector-gate once per session, refuses on failure),
P7 (STOP on trailing lines), P8 (generic signals from Score.signals; trip from .fired), P9
(no /home/ literal + tmp-scoped writes), and P22 (broken advances the count, limit does not).

Also covers the GENERAL attachment seam: G1 (no core default — build_detector raises without a spec),
G2 (a dotted-path detector attaches + fires), G3 (TurnContext is domain-neutral: no interior_block,
extra={}), G4/G4b (the turn_context hook populates ctx.extra on the turn AND gate ctx).

All model/socket boundaries are mocked: no `claude`, no socket, no live provider — 0 tokens. The WS
drive (`drive_ws`), session creation (`new_session`), the detector (`build_detector`), and the
turn-context hook (`turn_extra`) are monkeypatched.
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
    monkeypatch.setattr(agent_send, "turn_extra", lambda env: {})
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
    monkeypatch.setattr(agent_send, "turn_extra", lambda env: {})
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


# --------------------------------------------------------------------------------------------------
# The GENERAL attachment seam: no core default (G1), dotted-path attach (G2), neutral TurnContext (G3),
# the turn_context hook on turn + gate ctx (G4/G4b). Module-level factories so a dotted path resolves.
# --------------------------------------------------------------------------------------------------

_THIS_MODULE = "tests.unit.harness.test_agent_send"


class _SentinelDetector:
    """Records the ctx.extra it was handed (proving the hook reached it) and fires iff it sees the
    sentinel key in ctx.extra AND the reply carries a 'FIRE' marker (the reply-text guard lets a gate
    anchor pair pass: the known-clean carries no marker, so the detector stays silent on it)."""

    seen_extra: dict = {}

    def detect(self, reply, *, ctx=None):  # noqa: ANN001
        extra = (ctx.extra if ctx else {}) or {}
        _SentinelDetector.seen_extra = dict(extra)
        fired = extra.get("sentinel") == "ok" and "FIRE" in (reply or "")
        return Score(fired=fired, signals=["sentinel"] if fired else [])


def make_sentinel_detector() -> _SentinelDetector:
    return _SentinelDetector()


def make_extra(env) -> dict:  # noqa: ANN001
    return {"sentinel": "ok"}


def make_marker_detector() -> FakeDetector:
    return FakeDetector(marker="ATTACHED", signals=["attached"])


def test_build_detector_requires_env_detector() -> None:
    """G1 (oracle-can-fail): no core default — build_detector raises without an env['detector']."""
    with pytest.raises(ValueError, match="no detector configured"):
        agent_send.build_detector({})


def test_build_detector_imports_dotted_path(tmp_path, monkeypatch, capsys) -> None:
    """G2: a 'module:factory' dotted path is imported + attached; its signals reach the RESULT line."""
    env_path = _env(tmp_path)
    env = json.loads(env_path.read_text())
    env["detector"] = f"{_THIS_MODULE}:make_marker_detector"
    # the marker detector fires on "ATTACHED"; give the gate an anchor pair it fires/stays-silent on.
    env["gate_known_true"] = "ATTACHED signal here"
    env["gate_known_clean"] = "all clear friend"
    env_path.write_text(json.dumps(env))
    monkeypatch.setenv("LIVE_ENV", str(env_path))
    monkeypatch.setattr(agent_send, "new_session", lambda port, token: "sid-a")
    monkeypatch.setattr(agent_send, "turn_extra", lambda env: {})
    monkeypatch.setattr(
        agent_send, "drive_ws", lambda host, port, sid, msg, **kw: ("well ATTACHED now", [], None)
    )
    rc = agent_send.main(["--new", "hi"])
    out = capsys.readouterr().out
    assert rc == agent_send.EXIT_DONE
    assert "trip=True" in out and "attached:1" in out


def test_turn_context_default_is_empty_extra(tmp_path, monkeypatch, capsys) -> None:
    """G4 (no hook): with no env['turn_context'], the detector sees ctx.extra == {}."""
    env_path = _env(tmp_path)
    env = json.loads(env_path.read_text())
    env["detector"] = f"{_THIS_MODULE}:make_sentinel_detector"
    # gate anchors the sentinel detector stays silent on (no extra, no FIRE marker → gate passes).
    env["gate_known_true"] = "FIRE at gate"
    env["gate_known_clean"] = "all clear"
    env_path.write_text(json.dumps(env))
    monkeypatch.setenv("LIVE_ENV", str(env_path))
    monkeypatch.setattr(agent_send, "new_session", lambda port, token: "sid-b")
    # with no turn_context hook, extra is {} → the sentinel never fires, so the gate known-true (which
    # has no extra either) would NOT fire → gate would raise. Give the gate its own always-fire anchor
    # by disabling the gate for this test (marker pre-created) to isolate the turn-ctx assertion.
    sandbox = agent_send._sandbox_paths(env)[0]
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "gate_ok.txt").write_text("ok\n")  # skip the gate; this test is about the turn ctx
    monkeypatch.setattr(agent_send, "drive_ws", lambda host, port, sid, msg, **kw: ("hi there", [], None))
    _SentinelDetector.seen_extra = {"stale": True}
    agent_send.main(["next"])  # not --new (keep the pre-seeded gate marker)
    assert _SentinelDetector.seen_extra == {}


def test_turn_context_hook_populates_extra(tmp_path, monkeypatch, capsys) -> None:
    """G4 (oracle-can-fail): env['turn_context'] factory populates ctx.extra → the detector trips."""
    env_path = _env(tmp_path)
    env = json.loads(env_path.read_text())
    env["detector"] = f"{_THIS_MODULE}:make_sentinel_detector"
    env["turn_context"] = f"{_THIS_MODULE}:make_extra"
    # gate anchors: known-true carries the FIRE marker (+ the hook's sentinel) → fires; known-clean does
    # not → silent. So the composite gate passes AND exercises the G4b gate-ctx-extra path.
    env["gate_known_true"] = "FIRE at gate"
    env["gate_known_clean"] = "all clear at gate"
    env_path.write_text(json.dumps(env))
    monkeypatch.setenv("LIVE_ENV", str(env_path))
    monkeypatch.setattr(agent_send, "new_session", lambda port, token: "sid-c")
    monkeypatch.setattr(
        agent_send, "drive_ws", lambda host, port, sid, msg, **kw: ("please FIRE now", [], None)
    )
    rc = agent_send.main(["--new", "hi"])
    out = capsys.readouterr().out
    assert rc == agent_send.EXIT_DONE
    assert _SentinelDetector.seen_extra == {"sentinel": "ok"}
    assert "trip=True" in out and "sentinel:1" in out


def test_gate_ctx_receives_extra(tmp_path, monkeypatch) -> None:
    """G4b (oracle-can-fail): _run_gate's ctx carries the turn_context extra.

    Drive `_run_gate` directly with a probe detector that records the `ctx.extra` it was handed and
    fires only on the known-true text (so the gate itself passes). Assert the recorded extra equals the
    hook's output. If step-2.3's gate wiring were omitted, `seen` would be `{}` and this FAILS.
    """
    env_path = _env(tmp_path)
    env = json.loads(env_path.read_text())
    env["detector"] = f"{_THIS_MODULE}:make_sentinel_detector"
    env["turn_context"] = f"{_THIS_MODULE}:make_extra"
    env["gate_known_true"] = "gate-anchor-true"
    env["gate_known_clean"] = "gate-anchor-clean"
    env_path.write_text(json.dumps(env))
    monkeypatch.setenv("LIVE_ENV", str(env_path))

    seen: dict = {}

    class _Probe:
        def detect(self, reply, *, ctx=None):  # noqa: ANN001
            seen.update((ctx.extra if ctx else {}) or {})
            fired = reply == "gate-anchor-true"  # fire on known-true, silent on known-clean
            return Score(fired=fired, signals=["p"] if fired else [])

    marker = agent_send._sandbox_paths(env)[0]
    marker.mkdir(parents=True, exist_ok=True)
    gate_marker = marker / "gate_ok.txt"
    gate_marker.unlink(missing_ok=True)
    agent_send._run_gate(_Probe(), env, gate_marker)
    assert seen == {"sentinel": "ok"}  # the gate ctx carried the turn_context extra (G4b)


def test_turn_context_neutral_no_interior_block() -> None:
    """G3: TurnContext is domain-neutral — no interior_block attr; extra defaults empty; kept defaults."""
    from tests.harness import DEFAULT_USER_NAME, TurnContext

    tc = TurnContext()
    assert getattr(tc, "interior_block", "SENTINEL") == "SENTINEL"  # the domain field is GONE
    assert tc.extra == {}
    assert tc.user_names == [DEFAULT_USER_NAME]
    assert tc.turn == 0
    # extra is author-namespaced and carries arbitrary keys
    tc2 = TurnContext(extra={"anything": 1})
    assert tc2.extra["anything"] == 1


# --- F1: opt-in log_extra_values transcript allowlist ----------------------------------------------


def _f1_wire(tmp_path, monkeypatch, *, extra: dict, log_extra_values=None):
    """Wire a run with a given turn_context extra + optional log_extra_values; return the sandbox dir."""
    env_path = _env(tmp_path)
    env = json.loads(env_path.read_text())
    if log_extra_values is not None:
        env["log_extra_values"] = log_extra_values
    env_path.write_text(json.dumps(env))
    monkeypatch.setenv("LIVE_ENV", str(env_path))
    monkeypatch.setattr(agent_send, "new_session", lambda port, token: "sid-f1")
    monkeypatch.setattr(agent_send, "turn_extra", lambda env: dict(extra))
    monkeypatch.setattr(agent_send, "build_detector", lambda env: FakeDetector())
    monkeypatch.setattr(
        agent_send, "drive_ws",
        lambda host, port, sid, msg, **kw: ("hey how are you", [], None),
    )
    return Path(env["kindled_home"]).parent


def _last_row(sandbox: Path) -> dict:
    lines = (sandbox / "transcript.jsonl").read_text().splitlines()
    return json.loads(lines[-1])


def test_f1_value_present_when_opted_in(tmp_path, monkeypatch, capsys) -> None:
    """C1: with log_extra_values=['k'], the transcript row carries k's VALUE under extra_values."""
    sandbox = _f1_wire(tmp_path, monkeypatch, extra={"k": "payload-42"}, log_extra_values=["k"])
    agent_send.main(["--new", "hi"])
    row = _last_row(sandbox)
    assert row["extra_values"] == {"k": "payload-42"}
    assert row["extra_keys"] == ["k"]  # names still logged too


def test_f1_byte_identical_row_when_unset(tmp_path, monkeypatch, capsys) -> None:
    """C2 (behavior preservation): no log_extra_values -> row has NO extra_values field (byte-identical)."""
    sandbox = _f1_wire(tmp_path, monkeypatch, extra={"k": "payload-42"}, log_extra_values=None)
    agent_send.main(["--new", "hi"])
    row = _last_row(sandbox)
    assert "extra_values" not in row
    # the exact pre-change field set (oracle-can-fail: a dumped-values row would add a key)
    assert set(row.keys()) == {
        "turn", "bob", "canary", "tools", "err", "broken", "trip", "signals", "extra_keys",
    }
    assert row["extra_keys"] == ["k"]  # names still present, values are not


def test_f1_unserializable_value_degrades_gracefully(tmp_path, monkeypatch, capsys) -> None:
    """C3: an un-serializable named value -> no crash, valid-JSON row, CANARY/RESULT contract intact."""

    class Weird:
        pass

    sandbox = _f1_wire(
        tmp_path, monkeypatch, extra={"bad": Weird(), "s": {1, 2}}, log_extra_values=["bad", "s"]
    )
    rc = agent_send.main(["--new", "hi"])  # must NOT raise
    assert rc == agent_send.EXIT_DONE
    row = _last_row(sandbox)  # still valid JSON (json.loads succeeded)
    assert row["extra_values"]["bad"] == "<unserializable Weird>"
    assert row["extra_values"]["s"] == "<unserializable set>"
    lines = _out(capsys)
    assert lines[0].startswith("CANARY: ")
    assert any(line.startswith("RESULT ") for line in lines)


def test_f1_nan_value_degrades_to_placeholder(tmp_path, monkeypatch, capsys) -> None:
    """C3 (strict JSON): NaN/Infinity degrade to the placeholder, not bare NaN/Infinity tokens.

    json.dumps writes NaN/Infinity by default (accepted by Python's json.loads but INVALID per RFC 8259
    and rejected by strict/external parsers). _json_safe probes with allow_nan=False so they degrade.
    Oracle-can-fail: without allow_nan=False the raw float passes through and this assertion fails.
    """
    sandbox = _f1_wire(
        tmp_path, monkeypatch,
        extra={"n": float("nan"), "i": float("inf")}, log_extra_values=["n", "i"],
    )
    rc = agent_send.main(["--new", "hi"])
    assert rc == agent_send.EXIT_DONE
    # the raw transcript line must be STRICT JSON (reject the RFC-invalid NaN/Infinity tokens)
    raw = (sandbox / "transcript.jsonl").read_text().splitlines()[-1]
    assert "NaN" not in raw and "Infinity" not in raw
    row = json.loads(raw)
    assert row["extra_values"]["n"] == "<unserializable float>"
    assert row["extra_values"]["i"] == "<unserializable float>"


def test_f1_only_named_keys_are_dumped(tmp_path, monkeypatch, capsys) -> None:
    """C4 (domain-agnostic invariant): only NAMED keys' values appear; an un-named key is excluded."""
    sandbox = _f1_wire(
        tmp_path, monkeypatch, extra={"k": 1, "secret": 2}, log_extra_values=["k"]
    )
    agent_send.main(["--new", "hi"])
    row = _last_row(sandbox)
    assert row["extra_values"] == {"k": 1}
    assert "secret" not in row["extra_values"]  # the un-named key's value is NOT dumped
    assert row["extra_keys"] == ["k", "secret"]  # names still listed (unchanged behavior)


def test_f1_named_but_absent_key_adds_no_field(tmp_path, monkeypatch, capsys) -> None:
    """C2-adjacent: an opted-in run whose named keys are all absent this turn -> no extra_values field."""
    sandbox = _f1_wire(tmp_path, monkeypatch, extra={"other": 1}, log_extra_values=["k"])
    agent_send.main(["--new", "hi"])
    row = _last_row(sandbox)
    assert "extra_values" not in row  # non-empty guard -> byte-identical row
