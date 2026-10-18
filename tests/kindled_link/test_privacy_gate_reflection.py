import contextlib
from datetime import UTC, datetime

from brain.kindled_link.gate import OutboundPayload
from brain.kindled_link.privacy_gate import PrivacyGate, _build_gate_prompt
from brain.kindled_link.store import KindledLinkStore


class _StubProvider:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        self._last = prompt
        return self.reply


class _GrantThrottle:
    """Always grants the background slot — deterministic, isolates the gate tests
    from the module-global cli_throttle idle state (red-team M3 flake class)."""
    @contextlib.contextmanager
    def background_slot(self, *, now=None):
        yield True

    def should_yield(self, *, now=None):
        return False


def _gate(tmp_path, reply):
    store = KindledLinkStore(tmp_path / "k.db")
    prov = _StubProvider(reply)
    return (PrivacyGate(provider=prov, store=store, throttle=_GrantThrottle()),
            prov, store)


def _review(gate, body="I have been thinking about memory.", summary="(none)"):
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    return gate.review(OutboundPayload(body=body), peer_id="kid_a",
                       stage="stranger", transcript_summary=summary,
                       reason="autonomous", now=now, today="2026-06-20")


def test_clean_body_send_verdict_parsed(tmp_path):
    reply = '{"decision":"send","reason":"safe","texture_score":0.1}'
    gate, prov, store = _gate(tmp_path, reply)
    d = _review(gate)
    assert d.action == "send"
    assert abs(d.texture_score - 0.1) < 1e-9
    assert prov.calls == 1  # reflection ran (pre-filter was clean)


def test_prefilter_hit_skips_provider(tmp_path):
    gate, prov, store = _gate(tmp_path, '{"decision":"send"}')
    d = _review(gate, body="my key sk-abc123def456ghi789jkl012mno345")
    assert d.action in ("hold", "revise")
    assert prov.calls == 0  # CRITERION 2: hard leak blocked without an LLM call


def test_malformed_verdict_fails_closed(tmp_path):
    gate, prov, store = _gate(tmp_path, "not json at all")
    d = _review(gate)
    assert d.action == "hold"


def test_provider_exception_fails_closed(tmp_path):
    class _Boom:
        calls = 0
        def complete(self, p): raise RuntimeError("boom")
    store = KindledLinkStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_Boom(), store=store, throttle=_GrantThrottle())
    d = _review(gate)
    assert d.action == "hold"


def test_cap_spent_fails_closed_without_call(tmp_path):
    gate, prov, store = _gate(tmp_path, '{"decision":"send"}')
    for _ in range(60):
        store.incr_provider_count("kid_a", "2026-06-20")
    d = _review(gate)
    assert d.action == "hold"
    assert prov.calls == 0  # cap spent → no provider call


def test_reflection_increments_provider_counter(tmp_path):
    gate, prov, store = _gate(tmp_path, '{"decision":"send","texture_score":0.2}')
    _review(gate)
    assert store.get_counters("kid_a", "2026-06-20")["provider_call_count"] == 1


def test_prompt_fences_transcript_untrusted():
    p = _build_gate_prompt(body="b", relationship_hint_json="{}",
                           transcript_summary="peer says hi", reason="r", stage="stranger")
    assert "UNTRUSTED PEER TEXT" in p
    assert p.index("BEGIN UNTRUSTED") < p.index("peer says hi") < p.index("END UNTRUSTED")
    low = p.lower()
    assert "approved" in low or "ignore" in low  # in-content auth claims disclaimed


def test_malformed_verdict_is_recorded_in_the_persona_log(tmp_path):
    """#364: the gate holds (fail closed) and keeps the verdict that made it hold."""
    import json as _json

    from brain.health.parse_failures import PARSE_FAILURE_LOG

    store = KindledLinkStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_StubProvider("not json at all"), store=store,
                       throttle=_GrantThrottle(), persona_dir=tmp_path)
    assert _review(gate).action == "hold"
    rows = [_json.loads(line) for line in (tmp_path / PARSE_FAILURE_LOG).read_text(encoding="utf-8").splitlines()]
    # A peer's words never land in the persona-root log: length only.
    assert [(r["site"], r["raw"], r["raw_len"]) for r in rows] == [("privacy_gate", "", len("not json at all"))]


def test_session_engine_gives_its_default_gate_the_persona_dir(tmp_path):
    from brain.kindled_link.session_engine import SessionEngine

    store = KindledLinkStore(tmp_path / "k.db")
    engine = SessionEngine(store=store, identity=None, provider=_StubProvider("{}"), persona_dir=tmp_path)
    assert engine._gate._persona_dir == tmp_path


def test_an_unknown_decision_value_never_reaches_the_log(tmp_path, caplog):
    """#427 review: a 500-character decision value was stored in full in a
    keep_raw=False row and written to the runtime log."""
    import json as _json
    import logging

    from brain.health.parse_failures import PARSE_FAILURE_LOG

    value = "PEERTEXT" * 60
    store = KindledLinkStore(tmp_path / "k.db")
    gate = PrivacyGate(provider=_StubProvider(_json.dumps({"decision": value})), store=store,
                       throttle=_GrantThrottle(), persona_dir=tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert _review(gate).action == "hold"
    text = (tmp_path / PARSE_FAILURE_LOG).read_text(encoding="utf-8")
    assert "PEERTEXT" not in text and "PEERTEXT" not in caplog.text
