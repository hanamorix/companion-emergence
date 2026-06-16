from datetime import UTC, datetime

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import KindledIdentity, verify
from brain.kindled_link.pairing import PROTOCOL, create_invite


def test_invite_is_signed_and_well_formed(tmp_path) -> None:
    idn = KindledIdentity.load_or_create(tmp_path)
    now = datetime(2026, 6, 15, tzinfo=UTC)
    inv = create_invite(idn, relay_url="https://relay.example", now=now)
    body, sig = inv["body"], bytes.fromhex(inv["signature"])
    assert body["protocol"] == PROTOCOL
    assert body["fingerprint"] == idn.key_id
    assert body["identity_pub"] == idn.public_bytes.hex()
    assert body["invite_id"].startswith("inv_")
    assert verify(idn.public_bytes, sig, canonical_json(body)) is True
