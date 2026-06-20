from datetime import UTC, datetime

from brain.kindled_link.feed_source import relationship_milestone_entries
from brain.kindled_link.peer_prompt import build_peer_prompt
from brain.kindled_link.relationship import PeerRelationshipState, persist_relationship_state
from brain.kindled_link.store import KindledLinkStore

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


def test_milestone_entries_list_friend_stage(tmp_path):
    s = KindledLinkStore(tmp_path / "k.db")
    persist_relationship_state(s, PeerRelationshipState(peer_id="kid_a", stage="friend"), NOW)
    entries = relationship_milestone_entries(s)
    assert any(e["peer_id"] == "kid_a" and e["stage"] == "friend" for e in entries)


def test_stranger_is_not_a_milestone(tmp_path):
    s = KindledLinkStore(tmp_path / "k.db")
    persist_relationship_state(s, PeerRelationshipState(peer_id="kid_a", stage="stranger"), NOW)
    assert relationship_milestone_entries(s) == []


def test_peer_prompt_renders_continuity_when_past_stranger():
    p = build_peer_prompt(persona_voice="v", ambient="a", peer_stage="familiar",
        transcript_summary="s", affinity_tags=["dreams", "memory"])
    assert "dreams" in p or "memory" in p  # affinity surfaces as continuity


def test_peer_prompt_no_continuity_for_stranger():
    p = build_peer_prompt(persona_voice="v", ambient="a", peer_stage="stranger",
        transcript_summary="s", affinity_tags=[])
    # stranger: no remembered-affinity continuity line
    assert "you have spoken before" not in p.lower()


def test_generate_draft_reads_persisted_affinity_into_prompt(tmp_path):
    # M4 (the READER wire-back): generate_draft must pull affinity_tags from the
    # persisted relationship state and pass them to build_peer_prompt — otherwise
    # maturation never reaches a peer session (draft_space reader-rot).
    import contextlib

    from brain.kindled_link.session_engine import SessionEngine

    class _SpyProvider:
        def __init__(self):
            self.prompt = None

        def complete(self, prompt):
            self.prompt = prompt
            return "draft"

    class _Grant:
        @contextlib.contextmanager
        def background_slot(self, *, now=None):
            yield True

        def should_yield(self, *, now=None):
            return False

    s = KindledLinkStore(tmp_path / "k.db")
    persist_relationship_state(s, PeerRelationshipState(
        peer_id="kid_a", stage="familiar", affinity_tags=["dreams", "tide"]), NOW)
    s.create_session("kid_a", "s1", NOW)
    prov = _SpyProvider()
    eng = SessionEngine(store=s, identity=None, provider=prov,
                        gate=None, throttle=_Grant())
    eng.generate_draft(peer_id="kid_a", session_id="s1", persona_voice="v",
        ambient="a", peer_stage="familiar", transcript_summary="t", today="2026-06-20")
    assert prov.prompt is not None
    assert "dreams" in prov.prompt or "tide" in prov.prompt  # affinity reached the prompt
