from datetime import UTC, datetime

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.pairing import (
    confirm_local_fingerprint,
    create_invite,
    import_invite,
    mark_remote_paired,
)
from brain.kindled_link.store import KindledLinkStore


def test_two_persona_pairing_reaches_paired_both_sides(tmp_path) -> None:
    now = datetime(2026, 6, 15, tzinfo=UTC)
    idn_a = KindledIdentity.load_or_create(tmp_path / "A")
    idn_b = KindledIdentity.load_or_create(tmp_path / "B")
    store_a, store_b = KindledLinkStore(":memory:"), KindledLinkStore(":memory:")

    # A invites B; B invites A (mutual). No relay — invites handed across directly.
    inv_a = create_invite(idn_a, relay_url="https://r", now=now)
    inv_b = create_invite(idn_b, relay_url="https://r", now=now)
    import_invite(inv_b, store=store_a, now=now)  # A learns B -> pending_local
    import_invite(inv_a, store=store_b, now=now)  # B learns A -> pending_local

    # each user verifies the displayed phrase (matches the other's identity)
    confirm_local_fingerprint(store_a, idn_b.key_id, now=now)  # A: pending_remote
    confirm_local_fingerprint(store_b, idn_a.key_id, now=now)  # B: pending_remote

    # the simulator delivers each side's confirmation to the other
    mark_remote_paired(store_a, idn_b.key_id, now=now)  # A: paired
    mark_remote_paired(store_b, idn_a.key_id, now=now)  # B: paired

    assert store_a.get_peer(idn_b.key_id)["consent_state"] == "paired"
    assert store_b.get_peer(idn_a.key_id)["consent_state"] == "paired"
