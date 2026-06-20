import json
from datetime import UTC, datetime

from brain.kindled_link.gate import GateDecision
from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.store import KindledLinkStore


class _StubProvider:
    def complete(self, prompt): return "x"


class _CountingGate:
    def __init__(self): self.calls = 0
    def review(self, payload, **kw):
        self.calls += 1
        return GateDecision(action="hold")


def test_recover_regates_pending_draft_never_blind_resends(tmp_path):
    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    store.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="f",
                      consent_state="paired", relay_url="https://r", now=now)
    store.save_draft(peer_id="kid_a", session_id="s1",
                     payload_json=json.dumps({"body": "half-finished"}), now=now)

    gate = _CountingGate()
    eng = SessionEngine(store=store, identity=None, provider=_StubProvider(),
                        gate=gate)
    sent = []
    actions = eng.recover(now=now, today="2026-06-20", send_fn=lambda p: sent.append(p))

    assert gate.calls == 1            # the draft was RE-GATED, not blind-resent
    assert actions == ["hold"]
    assert sent == []                 # held → nothing left
    assert store.get_pending_drafts() == []  # draft resolved, off the pending list


def test_recover_defers_pacing_held_draft_keeps_it_pending(tmp_path):
    # re-red-team Major A: a draft not sendable for a PACING reason must stay
    # pending (retried later), NOT be dropped. Here last_outbound = now, so the
    # 60s gap is unmet at `now`.
    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.create_session("kid_a", "s1", now)
    store.bump_session_outbound("kid_a", "s1", now)  # last_outbound = now
    store.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="f",
                      consent_state="paired", relay_url="https://r", now=now)
    store.save_draft(peer_id="kid_a", session_id="s1",
                     payload_json=json.dumps({"body": "saved just before crash"}),
                     now=now)

    gate = _CountingGate()
    eng = SessionEngine(store=store, identity=None, provider=_StubProvider(),
                        gate=gate)
    actions = eng.recover(now=now, today="2026-06-20", send_fn=lambda p: None)

    assert actions == ["deferred"]
    assert gate.calls == 0                       # not re-gated yet — pacing not met
    assert len(store.get_pending_drafts()) == 1  # STILL pending, not dropped
