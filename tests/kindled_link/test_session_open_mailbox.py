"""build_session_open carries a signed sender_mailbox; parse returns it
(Phase 7a T2.4, piece 4 — leg-2 addressing)."""
from datetime import UTC, datetime, timedelta

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import build_session_open, parse_session_open

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def test_session_open_round_trips_sender_mailbox(tmp_path) -> None:
    idn = KindledIdentity.load_or_create(tmp_path)
    env = build_session_open(
        sender=idn, recipient_key_id="kid_r", relay_mailbox="mbx_recipient",
        session_id="s1", ephemeral_pub=b"\x01" * 32, bootstrap_nonce=b"\x02" * 16,
        sender_mailbox="mbx_sender", now=_NOW, ttl=timedelta(minutes=5),
    )
    assert env["sender_mailbox"] == "mbx_sender"
    parsed, reason = parse_session_open(env, sender_pub=idn.public_bytes, now=_NOW)
    assert reason is None
    assert parsed["sender_mailbox"] == "mbx_sender"
