from datetime import UTC, datetime

from brain.kindled_link.relationship import PeerRelationshipState, persist_relationship_state
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.views import holds_status, list_peers, peer_transcript

NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def _store(tmp_path):
    return KindledLinkStore(tmp_path / "k.db")


def test_list_peers_includes_stage_and_consent(tmp_path):
    s = _store(tmp_path)
    s.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="kid_a",
                  consent_state="paired", relay_url="https://r", now=NOW)
    persist_relationship_state(s, PeerRelationshipState(
        peer_id="kid_a", stage="familiar", affinity_tags=["dreams"]), NOW)
    peers = list_peers(s)
    assert len(peers) == 1
    p = peers[0]
    assert p["peer_id"] == "kid_a" and p["consent_state"] == "paired"
    assert p["stage"] == "familiar" and p["affinity_tags"] == ["dreams"]
    assert p["has_active_session"] is False


def test_peer_transcript_returns_messages(tmp_path):
    s = _store(tmp_path)
    s.append_transcript(peer_id="kid_a", session_id="s1", seq=1, direction="inbound",
                        text="peer said hi", now=NOW, provenance="peer")
    rows = peer_transcript(s, "kid_a")
    assert rows[0]["text"] == "peer said hi" and rows[0]["provenance"] == "peer"


def test_holds_status_never_exposes_draft_body(tmp_path):
    # THE SPINE: a held draft's body must never reach the holds projection.
    s = _store(tmp_path)
    s.save_draft(peer_id="kid_a", session_id="s1",
                 payload_json='{"body": "SECRET_USER_DETAIL_SENTINEL"}', now=NOW,
                 status="held")
    out = holds_status(s)
    assert out["held_count"] == 1
    assert out["items"][0]["session_id"] == "s1"
    assert "created_at" in out["items"][0]
    # the body sentinel must NOT appear anywhere in the projection
    import json as _j
    assert "SECRET_USER_DETAIL_SENTINEL" not in _j.dumps(out)
    assert "payload_json" not in _j.dumps(out)


def test_transcript_never_contains_a_held_draft_body(tmp_path):
    # m8: held drafts live in outbound_drafts, NEVER in transcript — so the
    # transcript view cannot leak a held body even if a hold exists.
    import json as _j
    s = _store(tmp_path)
    s.save_draft(peer_id="kid_a", session_id="s1",
                 payload_json='{"body": "SECRET_USER_DETAIL_SENTINEL"}', now=NOW,
                 status="held")
    rows = peer_transcript(s, "kid_a")
    assert "SECRET_USER_DETAIL_SENTINEL" not in _j.dumps(rows)
    assert rows == []  # nothing in the transcript table


def test_relay_health_reads_transport_log(tmp_path):
    """relay_health derives reachability from the transport audit log; fail-soft
    to relay_ok=None when no log exists (Phase 7a T12)."""
    from brain.kindled_link.audit import log_transport
    from brain.kindled_link.views import relay_health

    # No log yet → fail-soft.
    h0 = relay_health(tmp_path)
    assert h0["relay_ok"] is None
    assert h0["last_poll_ts"] is None
    assert h0["degraded_peers"] == []

    # A clean poll + push → relay_ok True.
    log_transport(tmp_path, event="poll", count=0, now=NOW)
    log_transport(tmp_path, event="push", peer_id="kid_a", now=NOW)
    h1 = relay_health(tmp_path)
    assert h1["relay_ok"] is True
    assert h1["last_poll_ts"] == NOW.isoformat()
    assert h1["last_push_ts"] == NOW.isoformat()

    # A flood_clamped marks the peer degraded; a relay_unavailable flips relay_ok.
    log_transport(tmp_path, event="flood_clamped", peer_id="kid_b", count=5, now=NOW)
    log_transport(tmp_path, event="relay_unavailable", now=NOW)
    h2 = relay_health(tmp_path)
    assert h2["relay_ok"] is False
    assert "kid_b" in h2["degraded_peers"]
