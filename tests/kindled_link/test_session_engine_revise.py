from datetime import UTC, datetime

from brain.kindled_link.gate import GateDecision, OutboundPayload
from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.store import KindledLinkStore


class _Prov:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return "revised body"


def _eng(tmp_path, gate, prov=None):
    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    eng = SessionEngine(store=store, identity=None, provider=prov or _Prov(),
                        gate=gate)
    return eng, store, now


class _ReviseThenSend:
    def __init__(self): self.n = 0
    def review(self, payload, **kw):
        self.n += 1
        if self.n == 1:
            return GateDecision(action="revise", revision_constraints="less")
        return GateDecision(action="send", texture_score=0.2)


class _AlwaysRevise:
    def review(self, payload, **kw):
        return GateDecision(action="revise", revision_constraints="less")


def test_revise_then_send_sends_revised_and_debits(tmp_path):
    sent = []
    eng, store, now = _eng(tmp_path, _ReviseThenSend())
    action = eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="orig"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: sent.append(p))
    assert action == "send"
    assert len(sent) == 1 and sent[0].body == "revised body"
    assert store.get_disclosure_budget("kid_a", now) < 1.0  # debited by 0.2


def test_second_revise_becomes_terminal_hold(tmp_path):
    sent = []
    eng, store, now = _eng(tmp_path, _AlwaysRevise())
    action = eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="orig"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: sent.append(p))
    assert action == "hold"  # at most one revision (parent §12)
    assert sent == []


def test_end_or_pause_ends_session_no_send(tmp_path):
    sent = []

    class _End:
        def review(self, payload, **kw): return GateDecision(action="end_or_pause")

    eng, store, now = _eng(tmp_path, _End())
    action = eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="x"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: sent.append(p))
    assert action == "end_or_pause"
    assert sent == []
    assert store.get_session("kid_a", "s1")["state"] == "ended"


def test_plain_send_debits_budget(tmp_path):
    sent = []

    class _Send:
        def review(self, payload, **kw): return GateDecision(action="send", texture_score=0.3)

    eng, store, now = _eng(tmp_path, _Send())
    eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="x"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: sent.append(p))
    assert abs(store.get_disclosure_budget("kid_a", now) - 0.7) < 1e-9


def test_revision_provider_call_counts_against_cap(tmp_path):
    # red-team M1: the revision provider.complete must increment the provider
    # counter (parent §9: draft + gate + revision all count against 60/day).
    prov = _Prov()  # counts complete() calls
    eng, store, now = _eng(tmp_path, _ReviseThenSend(), prov=prov)
    eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="orig"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: None)
    # one revision generation call was made AND counted
    assert prov.calls == 1
    assert store.get_counters("kid_a", "2026-06-20")["provider_call_count"] == 1


def test_revision_cap_spent_holds_without_call(tmp_path):
    # red-team M1/M2: if the provider cap is spent, _regenerate makes no call and
    # the draft is held (fail closed).
    prov = _Prov()
    eng, store, now = _eng(tmp_path, _ReviseThenSend(), prov=prov)
    for _ in range(60):
        store.incr_provider_count("kid_a", "2026-06-20")
    action = eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="orig"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: None)
    assert action == "hold"
    assert prov.calls == 0  # no revision call when cap spent


def test_revision_provider_error_fails_closed(tmp_path):
    # red-team M2: a provider exception during revision → hold, not a raw raise.
    class _BoomProv:
        calls = 0

        def complete(self, prompt):
            self.calls += 1
            raise RuntimeError("boom")

    eng, store, now = _eng(tmp_path, _ReviseThenSend(), prov=_BoomProv())
    action = eng.process_outbound(peer_id="kid_a", session_id="s1",
        payload=OutboundPayload(body="orig"), reason="r", now=now,
        today="2026-06-20", send_fn=lambda p: None)
    assert action == "hold"


def test_recovery_through_privacy_gate_holds_and_no_double_debit(tmp_path):
    # red-team M6: a recovered draft re-gated through the REAL PrivacyGate (default)
    # holds (empty transcript_summary → strict) and does not debit the budget.
    import json as _json

    from brain.kindled_link.privacy_gate import PrivacyGate

    class _HoldProv:
        def complete(self, prompt): return '{"decision":"hold"}'

    class _Grant:
        @__import__("contextlib").contextmanager
        def background_slot(self, *, now=None): yield True
        def should_yield(self, *, now=None): return False

    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    store.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="f",
                      consent_state="paired", relay_url="https://r", now=now)
    store.save_draft(peer_id="kid_a", session_id="s1",
                     payload_json=_json.dumps({"body": "recovered draft"}), now=now)
    eng = SessionEngine(store=store, identity=None, provider=_HoldProv(),
                        gate=PrivacyGate(provider=_HoldProv(), store=store,
                                         throttle=_Grant()))
    actions = eng.recover(now=now, today="2026-06-20", send_fn=lambda p: None)
    assert actions == ["hold"]
    assert store.get_disclosure_budget("kid_a", now) == 1.0  # no debit on a hold
