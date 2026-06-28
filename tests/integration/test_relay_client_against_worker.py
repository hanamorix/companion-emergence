"""End-to-end: the real RelayClient against the deployed CF Worker relay.

Skipped unless KINDLED_WORKER_URL is set (CI/dev opt-in) so the default suite
stays hermetic. The httpx client uses trust_env=False so a dev-shell proxy
(e.g. a local Anthropic proxy) doesn't intercept the call.

    KINDLED_WORKER_URL=https://kindled-relay.<sub>.workers.dev \
        uv run pytest tests/integration/test_relay_client_against_worker.py
"""
from __future__ import annotations

import os

import httpx
import pytest

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.relay_client import RelayClient

WORKER_URL = os.environ.get("KINDLED_WORKER_URL")
pytestmark = pytest.mark.skipif(not WORKER_URL, reason="set KINDLED_WORKER_URL to run")


def test_register_push_fetch_ack_roundtrip(tmp_path):
    idn = KindledIdentity.load_or_create(tmp_path)
    mailbox = "mbx_test_" + idn.key_id[-8:]
    with httpx.Client(base_url=WORKER_URL, timeout=15, trust_env=False) as http:
        client = RelayClient(http, identity=idn, mailbox_id=mailbox)
        client.register()
        env = {"relay_mailbox": mailbox, "sender_key_id": idn.key_id, "ciphertext": "deadbeef"}
        client.push(env)
        fetched = client.fetch()
        assert any(f["envelope"]["ciphertext"] == "deadbeef" for f in fetched)
        client.ack([fetched[0]["id"]])
        assert client.fetch() == []
