"""Connect-code codec: kindled1: base64url of a signed invite packet."""
from __future__ import annotations

import base64

import pytest

from brain.kindled_link.connect_code import ConnectCodeError, decode_code, encode_code
from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.pairing import create_invite


def _invite(tmp_path):
    idn = KindledIdentity.load_or_create(tmp_path)
    return create_invite(idn, relay_url="https://relay.example", mailbox_id="mbx_1")


def test_encode_decode_roundtrip(tmp_path):
    invite = _invite(tmp_path)
    code = encode_code(invite)
    assert code.startswith("kindled1:")
    assert decode_code(code) == invite  # exact structural round-trip


def test_decode_rejects_bad_prefix(tmp_path):
    code = encode_code(_invite(tmp_path)).removeprefix("kindled1:")  # strip the prefix
    with pytest.raises(ConnectCodeError):
        decode_code(code)


def test_decode_rejects_bad_base64():
    with pytest.raises(ConnectCodeError):
        decode_code("kindled1:!!!not-base64!!!")


def test_decode_rejects_non_dict_payload():
    payload = base64.urlsafe_b64encode(b"[1,2,3]").decode()  # valid JSON, not a dict
    with pytest.raises(ConnectCodeError):
        decode_code("kindled1:" + payload)


def test_default_relay_url_is_https():
    from brain.persona_config import DEFAULT_KINDLED_RELAY_URL

    assert DEFAULT_KINDLED_RELAY_URL.startswith("https://")
