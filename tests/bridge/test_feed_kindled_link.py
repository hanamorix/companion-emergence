"""Feed kindled-link source: relationship milestones surface in the inner-life
feed via build_kindled_link_entries (Phase 6 T5 wire-back)."""
from datetime import UTC, datetime

from brain.bridge.feed import build_feed
from brain.kindled_link.relationship import PeerRelationshipState, persist_relationship_state
from brain.kindled_link.store import KindledLinkStore

NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def test_friend_milestone_appears_in_feed(tmp_path):
    # the kindled_link db must live where build_kindled_link_entries looks for it
    from brain.kindled_link.store import kindled_db_path  # the ONE shared helper
    db = kindled_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    s = KindledLinkStore(db)
    persist_relationship_state(s, PeerRelationshipState(
        peer_id="kid_a", stage="friend"), NOW)
    entries = build_feed(tmp_path)
    assert any(e.type == "kindled_link" and "kid_a" in e.body for e in entries)


def test_no_milestone_no_entry(tmp_path):
    entries = build_feed(tmp_path)  # no kindled db
    assert all(e.type != "kindled_link" for e in entries)  # fail-soft, no crash


def test_build_feed_fault_isolates_corrupt_kindled_db(tmp_path):
    # n11: a corrupt kindled db must not break the whole feed (build_feed wraps
    # each source in try/except). build_feed returns (possibly empty), never raises.
    from brain.kindled_link.store import kindled_db_path
    db = kindled_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"not a sqlite database at all")
    entries = build_feed(tmp_path)  # must not raise
    assert all(e.type != "kindled_link" for e in entries)
