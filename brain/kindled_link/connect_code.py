"""Connect-code codec (design §5.1). A connect-code is the compact, pasteable
form of a signed invite packet (brain.kindled_link.pairing.create_invite):

    kindled1:<urlsafe-base64 of canonical_json(invite)>

The invite signature covers the whole body (incl. relay_url), so the code carries
the body verbatim — decode is a pure structural inverse; signature/expiry/consume
checks happen later in import_invite, not here.
"""
from __future__ import annotations

import base64
import json

from brain.kindled_link.codec import canonical_json

_PREFIX = "kindled1:"


class ConnectCodeError(ValueError):
    """A connect-code with a bad prefix, bad base64, or a non-dict payload."""


def encode_code(invite: dict) -> str:
    raw = canonical_json(invite)  # returns bytes already
    return _PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")


def decode_code(code: str) -> dict:
    if not isinstance(code, str) or not code.startswith(_PREFIX):
        raise ConnectCodeError("missing kindled1: prefix")
    b64 = code[len(_PREFIX):].strip()
    try:
        raw = base64.urlsafe_b64decode(b64.encode("ascii"))
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ConnectCodeError("malformed connect-code") from exc
    if not isinstance(payload, dict):
        raise ConnectCodeError("connect-code payload is not an object")
    return payload
