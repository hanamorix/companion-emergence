from datetime import UTC, datetime, timedelta

from brain.kindled_link import limits
from brain.kindled_link.store import KindledLinkStore


def _s(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_relationship_row_roundtrip(tmp_path):
    s = _s(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    assert s.get_relationship_row("kid_a") is None
    s.upsert_relationship_row(peer_id="kid_a", stage="acquaintance",
        trust_score=0.3, affinity_tags_json='["memory"]', boundaries_json="[]",
        repair_history_json="[]", evidence_json="[]", now=now)
    row = s.get_relationship_row("kid_a")
    assert row["stage"] == "acquaintance"
    assert row["trust_score"] == 0.3
    assert row["affinity_tags_json"] == '["memory"]'


def test_relationship_upsert_overwrites(tmp_path):
    s = _s(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.upsert_relationship_row(peer_id="kid_a", stage="stranger", trust_score=0.0,
        affinity_tags_json="[]", boundaries_json="[]", repair_history_json="[]",
        evidence_json="[]", now=now)
    s.upsert_relationship_row(peer_id="kid_a", stage="familiar", trust_score=0.6,
        affinity_tags_json="[]", boundaries_json="[]", repair_history_json="[]",
        evidence_json="[]", now=now)
    assert s.get_relationship_row("kid_a")["stage"] == "familiar"


def test_peer_emotion_accumulates_and_caps_window(tmp_path):
    s = _s(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    assert s.get_peer_emotion_accumulated("kid_a", now) == 0.0
    t1 = s.add_peer_emotion("kid_a", 0.2, now)
    assert abs(t1 - 0.2) < 1e-9
    t2 = s.add_peer_emotion("kid_a", 0.2, now)
    assert abs(t2 - 0.4) < 1e-9


def test_peer_emotion_window_decays(tmp_path):
    s = _s(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.add_peer_emotion("kid_a", 0.6, now)
    # after a full window the accumulator has fully decayed
    later = now + timedelta(hours=limits.PEER_EMOTION_WINDOW_HOURS)
    assert s.get_peer_emotion_accumulated("kid_a", later) == 0.0
