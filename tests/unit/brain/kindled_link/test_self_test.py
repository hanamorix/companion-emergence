"""Self-test orchestration against an in-process dev_relay (hermetic, no network)."""
from __future__ import annotations

from pathlib import Path

import httpx
from starlette.testclient import TestClient

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.self_test import run_self_test
from relay.dev_relay import create_app


def _relay_client() -> httpx.Client:
    # dev_relay with auth required (production posture). TestClient is an httpx.Client.
    return TestClient(create_app(require_auth=True), base_url="http://relay.test")


def test_self_test_all_stages_pass(tmp_path: Path):
    KindledIdentity.load_or_create(tmp_path)  # the real persona identity
    with _relay_client() as http:
        result = run_self_test(tmp_path, http=http)
    assert result["ok"] is True, result
    names = [s["name"] for s in result["stages"]]
    assert names == [
        "relay_reachable", "register", "pairing", "handshake",
        "message_sent", "message_received", "cleanup",
    ]
    assert all(s["ok"] for s in result["stages"]), result


def test_self_test_leaves_no_residue_in_real_store(tmp_path: Path):
    """The persona's real kindled_link.db must not exist / be empty of peers after a run."""
    from brain.kindled_link.store import KindledLinkStore, kindled_db_path

    KindledIdentity.load_or_create(tmp_path)
    with _relay_client() as http:
        run_self_test(tmp_path, http=http)
    db = kindled_db_path(tmp_path)
    if db.exists():
        store = KindledLinkStore(db, integrity_check=False)
        try:
            # No test peer should be persisted in the REAL store.
            assert store.list_paired_peers() == []
        finally:
            store.close()


def test_self_test_reports_relay_unreachable(tmp_path: Path):
    """A dead relay → relay_reachable ❌, run ok False, later stages short-circuit."""
    KindledIdentity.load_or_create(tmp_path)
    # A client pointed at a closed port-like transport that errors on every call.
    transport = httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ConnectError("boom")))
    with httpx.Client(base_url="http://dead.relay", transport=transport) as http:
        result = run_self_test(tmp_path, http=http)
    assert result["ok"] is False
    stages = {s["name"]: s for s in result["stages"]}
    assert stages["relay_reachable"]["ok"] is False
    # cleanup still runs
    assert stages["cleanup"]["ok"] is True
