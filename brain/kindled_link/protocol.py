"""kindled-link/1 wire protocol — canonical signing input, Ed25519 signatures,
X25519+HKDF session keys, ChaCha20-Poly1305 with a sequence-derived nonce, and
the receiver reject rules. Pure crypto + dict assembly; no I/O. See
docs/superpowers/specs/2026-06-15-kindled-to-kindled-protocol.md."""
from __future__ import annotations

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import KindledIdentity, verify

SUPPORTED_PROTOCOL = "kindled-link/1"

# Outer-envelope fields excluded from the AAD (bound everything else).
_AAD_EXCLUDE = ("ciphertext", "signature")
# Outer-envelope field excluded from the signing input.
_SIG_EXCLUDE = ("signature",)


def _without(outer: dict, keys) -> dict:
    return {k: v for k, v in outer.items() if k not in keys}


def aad_bytes(outer: dict) -> bytes:
    """Canonical AAD: the outer envelope minus ciphertext + signature."""
    return canonical_json(_without(outer, _AAD_EXCLUDE))


def sig_input_bytes(outer: dict) -> bytes:
    """Canonical signing input: the outer envelope minus the signature field
    (includes ciphertext)."""
    return canonical_json(_without(outer, _SIG_EXCLUDE))


def sign_envelope(outer: dict, idn: KindledIdentity) -> str:
    """Ed25519-sign sig_input_bytes(outer); return lowercase hex."""
    return idn.sign(sig_input_bytes(outer)).hex()


def verify_envelope_signature(outer: dict, sender_pub: bytes) -> bool:
    """Verify outer['signature'] (hex) over sig_input_bytes(outer). Fail-soft."""
    sig_hex = outer.get("signature")
    if not isinstance(sig_hex, str):
        return False
    try:
        sig = bytes.fromhex(sig_hex)
    except ValueError:
        return False
    return verify(sender_pub, sig, sig_input_bytes(outer))


def generate_ephemeral() -> X25519PrivateKey:
    """A fresh per-session X25519 ephemeral key pair."""
    return X25519PrivateKey.generate()


def derive_session_key(
    my_eph_priv: X25519PrivateKey,
    peer_eph_pub: bytes,
    *,
    sender_fp: str,
    recipient_fp: str,
    session_id: str,
    bootstrap_nonce: bytes,
) -> bytes:
    """HKDF-SHA256 over the ECDH shared secret (protocol §5). The info string
    sorts the two fingerprints, so both sides derive the identical key
    regardless of who initiated."""
    shared = my_eph_priv.exchange(X25519PublicKey.from_public_bytes(peer_eph_pub))
    fps = sorted([sender_fp, recipient_fp])
    info = b"kindled-link/1|" + fps[0].encode() + b"|" + fps[1].encode() + b"|" + session_id.encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=bootstrap_nonce, info=info).derive(shared)
