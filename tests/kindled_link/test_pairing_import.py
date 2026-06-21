from datetime import UTC, datetime, timedelta

import pytest

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.pairing import InviteError, create_invite, import_invite
from brain.kindled_link.store import KindledLinkStore


def _setup(tmp_path):
    idn_a = KindledIdentity.load_or_create(tmp_path / "A")
    store_b = KindledLinkStore(":memory:")
    now = datetime(2026, 6, 15, tzinfo=UTC)
    return idn_a, store_b, now


def test_import_stores_peer_pending_local(tmp_path) -> None:
    idn_a, store_b, now = _setup(tmp_path)
    inv = create_invite(idn_a, relay_url="https://r", now=now)
    result = import_invite(inv, store=store_b, now=now)
    assert result["peer_id"] == idn_a.key_id
    assert result["fingerprint_phrase"]  # non-empty
    peer = store_b.get_peer(idn_a.key_id)
    assert peer["consent_state"] == "pending_local"


def test_invite_carries_mailbox_into_peer_row(tmp_path) -> None:
    """An invite created with a mailbox_id round-trips to peers.relay_mailbox."""
    idn_a, store_b, now = _setup(tmp_path)
    inv = create_invite(idn_a, relay_url="https://r", mailbox_id="mbx_abc123", now=now)
    import_invite(inv, store=store_b, now=now)
    assert store_b.get_peer(idn_a.key_id)["relay_mailbox"] == "mbx_abc123"


def test_bad_signature_rejected(tmp_path) -> None:
    idn_a, store_b, now = _setup(tmp_path)
    inv = create_invite(idn_a, relay_url="https://r", now=now)
    inv["body"]["relay_url"] = "https://evil"  # body changed → signature invalid
    with pytest.raises(InviteError, match="bad_signature"):
        import_invite(inv, store=store_b, now=now)


def test_fingerprint_mismatch_rejected(tmp_path) -> None:
    idn_a, store_b, now = _setup(tmp_path)
    inv = create_invite(idn_a, relay_url="https://r", now=now)
    inv["body"]["fingerprint"] = "kid_0000000000000000"
    # re-sign so the signature is valid but fingerprint != hash(pub)
    inv["signature"] = idn_a.sign(canonical_json(inv["body"])).hex()
    with pytest.raises(InviteError, match="fingerprint_mismatch"):
        import_invite(inv, store=store_b, now=now)


def test_expired_invite_rejected(tmp_path) -> None:
    idn_a, store_b, now = _setup(tmp_path)
    inv = create_invite(idn_a, relay_url="https://r", now=now)
    later = now + timedelta(days=8)
    with pytest.raises(InviteError, match="expired"):
        import_invite(inv, store=store_b, now=later)


def test_single_use_invite_rejected_on_second_import(tmp_path) -> None:
    idn_a, store_b, now = _setup(tmp_path)
    inv = create_invite(idn_a, relay_url="https://r", now=now)
    import_invite(inv, store=store_b, now=now)
    with pytest.raises(InviteError, match="invite_consumed"):
        import_invite(inv, store=store_b, now=now)
