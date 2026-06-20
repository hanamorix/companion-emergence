from datetime import UTC, datetime

from brain.kindled_link.privacy_gate import PrivacyGate
from brain.kindled_link.session_engine import SessionEngine
from brain.kindled_link.store import KindledLinkStore


class _StubProvider:
    def complete(self, prompt):
        return '{"decision":"hold"}'


def test_default_gate_is_privacy_gate(tmp_path):
    store = KindledLinkStore(tmp_path / "k.db")
    eng = SessionEngine(store=store, identity=None, provider=_StubProvider())
    assert isinstance(eng._gate, PrivacyGate)


def test_process_outbound_passes_stranger_stage_not_consent_state(tmp_path):
    # the gate must receive stage='stranger', never the peer's consent_state.
    seen = {}

    class _SpyGate:
        def review(self, payload, **kw):
            seen.update(kw)
            from brain.kindled_link.gate import GateDecision
            return GateDecision(action="hold")

    store = KindledLinkStore(tmp_path / "k.db")
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    store.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="f",
                      consent_state="paired", relay_url="https://r", now=now)
    store.create_session("kid_a", "s1", now)
    eng = SessionEngine(store=store, identity=None, provider=_StubProvider(),
                        gate=_SpyGate())
    eng.process_outbound(peer_id="kid_a", session_id="s1",
                         payload=__import__("brain.kindled_link.gate", fromlist=["OutboundPayload"]).OutboundPayload(body="hi"),
                         reason="r", now=now, today="2026-06-20", send_fn=lambda p: None)
    assert seen["stage"] == "stranger"  # NOT "paired"
