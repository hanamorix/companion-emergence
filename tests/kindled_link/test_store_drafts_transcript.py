from datetime import UTC, datetime

from brain.kindled_link.store import KindledLinkStore


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_pending_draft_roundtrip(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    did = s.save_draft(peer_id="kid_a", session_id="sess1",
                       payload_json='{"body":"hi"}', now=now)
    pending = s.get_pending_drafts()
    assert len(pending) == 1
    assert pending[0]["id"] == did
    assert pending[0]["payload_json"] == '{"body":"hi"}'


def test_set_draft_status_removes_from_pending(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    did = s.save_draft(peer_id="kid_a", session_id="sess1",
                       payload_json="{}", now=now)
    s.set_draft_status(did, "held")
    assert s.get_pending_drafts() == []


def test_count_holds_for_peer(tmp_path):
    # Encapsulates the 'hold'-status count (was a direct store._conn reach in
    # tick._count_recent_holds). Counts only this peer's hold-status drafts.
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    d1 = s.save_draft(peer_id="kid_a", session_id="s1", payload_json="{}", now=now)
    d2 = s.save_draft(peer_id="kid_a", session_id="s1", payload_json="{}", now=now)
    d3 = s.save_draft(peer_id="kid_b", session_id="s2", payload_json="{}", now=now)
    s.set_draft_status(d1, "hold")
    s.set_draft_status(d2, "send")  # not a hold
    s.set_draft_status(d3, "hold")  # different peer
    assert s.count_holds_for_peer("kid_a") == 1
    assert s.count_holds_for_peer("kid_b") == 1
    assert s.count_holds_for_peer("kid_c") == 0


def test_transcript_provenance_and_order(tmp_path):
    s = _store(tmp_path)
    now = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    s.append_transcript(peer_id="kid_a", session_id="sess1", seq=1,
                        direction="inbound", text="peer said hi", now=now,
                        provenance="peer")
    s.append_transcript(peer_id="kid_a", session_id="sess1", seq=2,
                        direction="outbound", text="held draft", now=now,
                        provenance="local")
    rows = s.recent_transcript("kid_a", limit=10)
    assert rows[0]["seq"] == 2  # newest first
    assert rows[1]["provenance"] == "peer"
