"""Bug #6: peer key rotation must rekey peer_id (the PK == fingerprint) across
ALL peer-keyed tables, or post-rotation lookups (by the peer's NEW key id) miss
and correspondence breaks permanently."""
from datetime import UTC, datetime

from brain.kindled_link.store import KindledLinkStore


def _store():
    return KindledLinkStore(":memory:")


def test_update_peer_identity_rekeys_peer_id_and_child_rows():
    s = _store()
    now = datetime(2026, 7, 2, tzinfo=UTC)
    old = "kid_old"
    new = "kid_new"
    s.upsert_peer(peer_id=old, identity_pub_hex="aa" * 32, fingerprint=old,
                  consent_state="paired", relay_url="https://r", now=now)
    # Child rows keyed on peer_id.
    s.append_transcript(peer_id=old, session_id="s1", seq=1, direction="inbound",
                        text="hi", now=now, provenance="peer")
    s.set_seq_high_water(old, "s1", 5)

    # Rotate: new pubkey → new fingerprint (new_key_id).
    s.update_peer_identity(old, "bb" * 32, new, now)

    # Peer row now resolvable by the NEW key id, not the old.
    p = s.get_peer(new)
    assert p is not None, "peer lost after rotation — get_peer(new_fingerprint) missed"
    assert p["identity_pub"] == "bb" * 32
    assert p["previous_identity_pub"] == "aa" * 32
    assert s.get_peer(old) is None, "stale old-keyed row still present"

    # Child rows moved with the peer.
    assert s.get_seq_high_water(new, "s1") == 5
    assert [r["text"] for r in s.recent_transcript(new)] == ["hi"]
    assert s.get_seq_high_water(old, "s1") == 0  # nothing left under the old id
