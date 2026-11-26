"""#481: a bounded retry loop must not sleep after the LAST failed attempt (nothing follows it, the caller only waits).

`RelayClient._post` slept 2 s after its third transport failure before raising, and `_allocate_port` slept 1 s after its third
failed bind before raising. Both now sleep only when another attempt follows.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from brain.bridge import runner
from brain.kindled_link import relay_client
from brain.kindled_link.relay_client import RelayClient, RelayUnavailableError


def test_the_relay_client_sleeps_between_attempts_but_not_after_the_last(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(relay_client.time, "sleep", sleeps.append)

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://relay.example")
    client = RelayClient(http, identity=object(), mailbox_id="m1")

    with pytest.raises(RelayUnavailableError):
        client._post("/x", {})

    assert sleeps == [0.5, 1.0]  # before attempts 2 and 3; nothing after the third failure


def test_the_relay_client_still_returns_after_a_transient_failure(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(relay_client.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"ok": True})

    http = httpx.Client(transport=httpx.MockTransport(flaky), base_url="http://relay.example")

    resp = RelayClient(http, identity=object(), mailbox_id="m1")._post("/x", {})

    assert resp.status_code == 200 and sleeps == [0.5]  # PIN: the retry still backs off once


def test_allocate_port_sleeps_between_attempts_but_not_after_the_last(monkeypatch):
    import time

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    class Refusing:
        def __init__(self, *a, **k):
            pass

        def bind(self, addr):
            raise OSError("no free port")

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", Refusing)

    with pytest.raises(RuntimeError, match="failed to bind a free port"):
        runner._allocate_port()

    assert sleeps == [0.001, 0.01]  # the 0.1 s that used to follow the last failure is gone


def test_allocate_port_returns_a_port_without_sleeping():
    # PIN
    port = runner._allocate_port()

    assert 0 < port < 65536
