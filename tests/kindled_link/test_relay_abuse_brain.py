"""#48 brain-side fail-soft: relay_client raises on 4xx (C-12); poll_and_ingest
guards the fetch so a relay rejection degrades cleanly (C-12b)."""
from datetime import UTC, datetime

import pytest

from brain.kindled_link.relay_client import RelayClient, RelayUnavailableError


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, resp):
        self._resp = resp

    def post(self, path, json=None):
        return self._resp


class _Idn:
    public_bytes = b"\x00" * 32

    def sign(self, body):
        return b"\x01" * 64


def test_c12_post_raises_on_4xx():
    # a 429 (or any >=400) from the relay must raise RelayUnavailableError, not
    # return a Response whose .json()["nonce"] KeyErrors.
    client = RelayClient(_FakeHttp(_Resp(429)), identity=_Idn(), mailbox_id="m")
    with pytest.raises(RelayUnavailableError):
        client.push({"relay_mailbox": "m"})


def test_c12_challenge_4xx_raises_not_keyerror():
    client = RelayClient(_FakeHttp(_Resp(404, {"detail": "nope"})), identity=_Idn(), mailbox_id="m")
    with pytest.raises(RelayUnavailableError):
        client.fetch()  # fetch -> _auth -> /mailbox/challenge 404


def test_c12_2xx_still_returns_normally():
    client = RelayClient(_FakeHttp(_Resp(200, {"id": "env_1"})), identity=_Idn(), mailbox_id="m")
    client.push({"relay_mailbox": "m"})  # no raise


def test_c12b_poll_and_ingest_degrades_on_relay_error(tmp_path):
    # poll_and_ingest must NOT propagate RelayUnavailableError — it returns a
    # degraded summary so the tick completes + saves cadence.
    from brain.kindled_link.identity import KindledIdentity
    from brain.kindled_link.store import KindledLinkStore
    from brain.kindled_link.transport import poll_and_ingest

    class _RaisingRelay:
        def fetch(self):
            raise RelayUnavailableError("relay /mailbox/fetch -> 429")

    store = KindledLinkStore(tmp_path / "k.db")
    idn = KindledIdentity.load_or_create(tmp_path)
    out = poll_and_ingest(store, idn, _RaisingRelay(),
                          now=datetime(2026, 6, 24, 12, 0, tzinfo=UTC))
    assert isinstance(out, dict)  # returned cleanly, no raise
