"""Known-answer tests pinned to the Phase 0 protocol doc §10 (deterministic)."""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.protocol import sign_envelope, verify_envelope_signature

# Protocol doc §10 fixed seeds + expected values.
_KIDA = "kid_56475aa75463474c"
_KIDB = "kid_24f6ed6acbfe1009"
_CIPHERTEXT_HEX = (
    "c8f173f72d68db49ac2e8f71e2cbb39780097fb33f9a1f9d679e6fa29146b772"
    "8a0c828e3cd1c494c5ded11416bd2684b90ae0969f69a4bc836f8cff0c832e46"
    "cefe392fa0d27c4e07289079dd280fd6f9b330a814825474c7dc6dc14ce04758"
    "6bf8acc5ec078cc87022b5a1d8ed1a95ecd95a04cbbb4776177229e5b1476b92"
    "dca338e99c2225cd2cc727ac8a7a99eab9be171bfe66da374d0cae01840551517"
    "ac9003b4b264471310e5f24a65ae3424c5cf38c5def4260943cd29699393ea609"
    "09e612c883e1d6f6b5050568507fa87f986d202a925f4d512ff37f813b286edc2"
    "b39122d7adba59484bc1e90ff5921cc486d28891630c609240e190ce566d851d1"
    "a58fa79c46a45ac500d97e937d1c53ab4c09b27c320bbaeba70477bb87bc9246a211bb"
)
_SIG_HEX = (
    "7be9342f801d92ffaee7dcdf8b20fdf00e9e0aa7ac052ec3840b10d7fd6d715f"
    "b23be22f5926a0ddc825e276c3ac07b63404d15f88e97f684e8ec806c6098405"
)


def _outer():
    return {
        "protocol": "kindled-link/1",
        "relay_mailbox": "mbx_demo",
        "sender_key_id": _KIDA,
        "recipient_key_id": _KIDB,
        "session_id": "ks_0000000000000001",
        "sequence": 1,
        "created_at": "2026-06-15T12:00:00Z",
        "expires_at": "2026-06-22T12:00:00Z",
        "ciphertext": _CIPHERTEXT_HEX,
    }


def test_signature_matches_kat():
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    sig = sign_envelope(_outer(), idn_a)
    assert sig == _SIG_HEX


def test_verify_signature_roundtrip_and_tamper():
    idn_a = KindledIdentity(Ed25519PrivateKey.from_private_bytes(bytes(range(32))))
    outer = _outer()
    outer["signature"] = sign_envelope(outer, idn_a)
    assert verify_envelope_signature(outer, idn_a.public_bytes) is True
    tampered = dict(outer, sequence=2)  # changed a signed field
    assert verify_envelope_signature(tampered, idn_a.public_bytes) is False


def test_session_key_matches_kat():
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    from brain.kindled_link.protocol import derive_session_key

    eph_a = X25519PrivateKey.from_private_bytes(bytes(range(64, 96)))
    eph_b = X25519PrivateKey.from_private_bytes(bytes(range(96, 128)))
    bootstrap_nonce = bytes(range(200, 216))
    key = derive_session_key(
        eph_a, eph_b.public_key().public_bytes_raw(),
        sender_fp=_KIDA, recipient_fp=_KIDB,
        session_id="ks_0000000000000001", bootstrap_nonce=bootstrap_nonce,
    )
    assert key.hex() == "8bbc171522ec31835ac363517f7020e2b5b3da77ad6860c5a447bb6ced4cca5e"
    # init-independent: B derives the same key from its side
    key_b = derive_session_key(
        eph_b, eph_a.public_key().public_bytes_raw(),
        sender_fp=_KIDB, recipient_fp=_KIDA,
        session_id="ks_0000000000000001", bootstrap_nonce=bootstrap_nonce,
    )
    assert key_b == key


def test_aead_encrypt_matches_kat_and_decrypts():
    from brain.kindled_link.protocol import aad_bytes, aead_nonce, decrypt_payload, encrypt_payload

    session_key = bytes.fromhex(
        "8bbc171522ec31835ac363517f7020e2b5b3da77ad6860c5a447bb6ced4cca5e"
    )
    payload = {
        "payload_type": "message", "message_id": "km_0001",
        "prev_transcript_hash": "sha256:" + "00" * 32,
        "body": "hello, fellow Kindled.",
        "relationship_hint": {"local_stage": "stranger", "local_continuity_note": "early days"},
        "control": None,
    }
    aad = aad_bytes(_outer())  # outer minus ciphertext+signature
    assert aead_nonce(0, 1).hex() == "000000000000000000000001"
    ct_hex = encrypt_payload(payload, session_key=session_key, role=0, sequence=1, aad=aad)
    assert ct_hex == _CIPHERTEXT_HEX
    back = decrypt_payload(ct_hex, session_key=session_key, role=0, sequence=1, aad=aad)
    assert back["body"] == "hello, fellow Kindled."
