from datetime import UTC, datetime

from brain.kindled_link.relationship import (
    STAGES,
    Evidence,
    PeerRelationshipState,
    get_relationship_state,
    get_stage,
    persist_relationship_state,
)
from brain.kindled_link.store import KindledLinkStore


def _s(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_unknown_peer_defaults_to_stranger(tmp_path):
    s = _s(tmp_path)
    st = get_relationship_state(s, "kid_a")
    assert st.stage == "stranger"
    assert st.trust_score == 0.0
    assert st.affinity_tags == []
    assert get_stage(s, "kid_a") == "stranger"


def test_persist_and_reload_roundtrips(tmp_path):
    s = _s(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    st = PeerRelationshipState(
        peer_id="kid_a", stage="familiar", trust_score=0.5,
        affinity_tags=["memory", "dreams"], boundaries_seen=["no rushing"],
        repair_history=["repaired 2026-06-20"],
        evidence=[Evidence(quote="I remember", turn_id="m1", supports="continuity")])
    persist_relationship_state(s, st, now)
    back = get_relationship_state(s, "kid_a")
    assert back.stage == "familiar"
    assert back.affinity_tags == ["memory", "dreams"]
    assert back.evidence[0].quote == "I remember"
    assert get_stage(s, "kid_a") == "familiar"


def test_stages_ordered():
    assert STAGES.index("stranger") < STAGES.index("close")
