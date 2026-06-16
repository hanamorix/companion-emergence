"""kindled-link/1 wire protocol — canonical signing input, Ed25519 signatures,
X25519+HKDF session keys, ChaCha20-Poly1305 with a sequence-derived nonce, and
the receiver reject rules. Pure crypto + dict assembly; no I/O. See
docs/superpowers/specs/2026-06-15-kindled-to-kindled-protocol.md."""
from __future__ import annotations

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
