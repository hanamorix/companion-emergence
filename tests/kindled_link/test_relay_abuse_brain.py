"""#48 brain-side fail-soft: relay_client raises on 4xx (C-12); poll_and_ingest
guards the fetch so a relay rejection degrades cleanly (C-12b)."""
from datetime import UTC, datetime

import httpx
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


def test_c12b_relay_error_text_reaches_the_transport_log(tmp_path):
    """#367: a wrong relay URL or a 429 used to read as an empty mailbox; the
    transport row now says why."""
    import json

    from brain.kindled_link.identity import KindledIdentity
    from brain.kindled_link.store import KindledLinkStore
    from brain.kindled_link.transport import poll_and_ingest

    class _RaisingRelay:
        def fetch(self):
            raise RelayUnavailableError("relay /mailbox/fetch -> 429")

    store = KindledLinkStore(tmp_path / "k.db")
    idn = KindledIdentity.load_or_create(tmp_path)
    poll_and_ingest(store, idn, _RaisingRelay(), now=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
                    persona_dir=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "kindled_link" / "transport.jsonl").read_text(encoding="utf-8").splitlines()]
    (row,) = [r for r in rows if r["event"] == "poll_relay_unavailable"]
    assert row["error"] == "relay /mailbox/fetch -> 429"



def test_send_relay_error_text_reaches_the_transport_log(tmp_path, monkeypatch):
    import json

    import brain.kindled_link.transport as transport

    class _Store:
        def get_session_key(self, peer_id, session_id):
            return {"my_role": "initiator", "session_key": b"k" * 32}

        def next_outbound_sequence(self, peer_id, session_id):
            return 1

        def get_peer(self, peer_id):
            return {"relay_mailbox": "mb"}

    class _RaisingRelay:
        def push(self, env):
            raise RelayUnavailableError("relay /envelope -> 503")

    monkeypatch.setattr(transport, "build_envelope", lambda **kw: {"env": 1})
    ok = transport.send_message(_Store(), object(), _RaisingRelay(), peer_id="p", session_id="s",
                                payload={}, now=datetime(2026, 6, 24, 12, 0, tzinfo=UTC), persona_dir=tmp_path)
    assert ok is False
    rows = [json.loads(line) for line in (tmp_path / "kindled_link" / "transport.jsonl").read_text(encoding="utf-8").splitlines()]
    (row,) = [r for r in rows if r["event"] == "relay_unavailable"]
    assert row["error"] == "relay /envelope -> 503"


def test_queued_rotation_notice_keeps_the_relay_error(tmp_path):
    """#367: the notice stays queued (as before) and the transport row says why."""
    import json

    from brain.kindled_link.tick import _drain_pending_rotation_notices

    class _Store:
        def pop_pending_rotation_notices(self):
            env = {"expires_at": "2099-01-01T00:00:00Z"}
            return [{"peer_id": "p", "envelope_json": json.dumps(env)}]

        def clear_rotation_notice(self, peer_id):
            raise AssertionError("a relay failure must leave the notice queued")

    class _RaisingRelay:
        def push(self, env):
            raise RelayUnavailableError("relay /envelope -> 429")

    _drain_pending_rotation_notices(_Store(), _RaisingRelay(), persona_dir=tmp_path,
                                   now=datetime(2026, 6, 24, 12, 0, tzinfo=UTC))
    rows = [json.loads(line) for line in (tmp_path / "kindled_link" / "transport.jsonl").read_text(encoding="utf-8").splitlines()]
    (row,) = [r for r in rows if r["event"] == "rotation_notice_queued"]
    assert row["error"] == "relay /envelope -> 429"


def test_a_transport_error_with_no_message_still_names_itself(monkeypatch):
    """#429 review: str(httpx error) can be empty, which logged `error: ""`."""
    import httpx

    from brain.kindled_link.identity import KindledIdentity
    from brain.kindled_link.relay_client import RelayClient

    def boom(request):
        raise httpx.ConnectError("")

    client = httpx.Client(transport=httpx.MockTransport(boom), base_url="http://relay.test")
    rc = RelayClient(client, identity=KindledIdentity.__new__(KindledIdentity), mailbox_id="mb")
    import brain.kindled_link.relay_client as relay_client_mod

    monkeypatch.setattr(relay_client_mod.time, "sleep", lambda s: None)  # no backoff wait
    with pytest.raises(RelayUnavailableError) as info:
        rc.push({"env": 1})
    assert str(info.value) == "ConnectError"


# #442: the relay is untrusted, so a 2xx reply with a bad body, or an httpx
# error that is not a transport error, must also surface as
# RelayUnavailableError (the only exception every caller catches).

def _mock_client(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://relay.test")
    return RelayClient(http, identity=_Idn(), mailbox_id="m")


@pytest.mark.parametrize(
    "challenge",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json={"detail": "no nonce here"}),
        httpx.Response(200, json=["nonce"]),
    ],
    ids=["non-json", "missing-key", "not-an-object"],
)
def test_442_malformed_2xx_challenge_raises_relay_unavailable(challenge):
    client = _mock_client(lambda request: challenge)
    with pytest.raises(RelayUnavailableError, match="malformed reply"):
        client.fetch()


@pytest.mark.parametrize(
    "fetched",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"nonce": "n"}),
        httpx.Response(200, json=None),
    ],
    ids=["non-json", "missing-key", "null-body"],
)
def test_442_malformed_2xx_fetch_raises_relay_unavailable(fetched):
    def handler(request):
        if request.url.path == "/mailbox/challenge":
            return httpx.Response(200, json={"nonce": "n"})
        return fetched

    with pytest.raises(RelayUnavailableError, match="malformed reply"):
        _mock_client(handler).fetch()


@pytest.mark.parametrize("exc_type", [httpx.DecodingError, httpx.TooManyRedirects])
def test_442_non_transport_request_error_raises_relay_unavailable(exc_type, monkeypatch):
    sleeps = []
    monkeypatch.setattr("brain.kindled_link.relay_client.time.sleep", sleeps.append)

    def handler(request):
        raise exc_type("boom", request=request)

    with pytest.raises(RelayUnavailableError):
        _mock_client(handler).push({"relay_mailbox": "m"})
    assert sleeps == [], "a non-transport error is not transient, so it is not retried"
