"""In-process self-test (design §6): drive the full Kindled-link loop — relay →
pair → handshake → encrypted round-trip — between the persona's real identity and
a throwaway ephemeral peer, each on its own TEMP store, reporting per-stage results.

Zero residue: both sides use temp KindledLinkStores in tmp dirs (the persona's real
kindled_link.db is never opened); cleanup drops the tmp dirs + best-effort drains the
temp mailboxes. Fixed canned message — no LLM, no tokens.
"""
from __future__ import annotations

import contextlib
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from brain.kindled_link.identity import KindledIdentity
from brain.kindled_link.relay_client import RelayClient, RelayUnavailableError
from brain.kindled_link.session import open_session
from brain.kindled_link.store import KindledLinkStore
from brain.kindled_link.transport import poll_and_ingest, send_message

_CANNED = {"text": "kindled self-test ✓"}


def run_self_test(persona_dir: Path, *, http: httpx.Client, now: datetime | None = None) -> dict:
    """Drive the full Kindled-link loop between the persona's real identity and a
    throwaway ephemeral peer, both on temp stores. Never raises. Returns per-stage
    contract: {"ok": bool, "stages": [...], "relay_url": str}."""
    now = now or datetime.now(UTC)
    stages: list[dict] = []

    def stage(name: str, ok: bool, detail: str = "") -> None:
        stages.append({"name": name, "ok": ok, "detail": detail})

    def result(ok: bool) -> dict:
        return {"ok": ok, "stages": stages, "relay_url": str(http.base_url)}

    a_dir = Path(tempfile.mkdtemp(prefix="kself-a-"))
    b_dir = Path(tempfile.mkdtemp(prefix="kself-b-"))
    a_store: KindledLinkStore | None = None
    b_store: KindledLinkStore | None = None
    a_client: RelayClient | None = None
    b_client: RelayClient | None = None

    try:
        # Persona identity (real, read-only) on a TEMP store; ephemeral peer identity.
        a_idn = KindledIdentity.load_or_create(persona_dir)
        b_idn = KindledIdentity.load_or_create(b_dir)
        a_store = KindledLinkStore(a_dir / "k.db")
        b_store = KindledLinkStore(b_dir / "k.db")
        a_mbx = a_store.get_or_create_local_mailbox()
        b_mbx = b_store.get_or_create_local_mailbox()
        a_client = RelayClient(http, identity=a_idn, mailbox_id=a_mbx)
        b_client = RelayClient(http, identity=b_idn, mailbox_id=b_mbx)

        # 1. relay_reachable + 2. register (folded: a register that connects = reachable)
        try:
            a_client.register()
            b_client.register()
            stage("relay_reachable", True)
            stage("register", True)
        except (RelayUnavailableError, httpx.HTTPError, httpx.TransportError) as exc:
            stage("relay_reachable", False, f"relay unreachable: {exc}")
            return result(False)

        # 3. pairing (in-process: each side upserts the other as paired)
        try:
            a_store.upsert_peer(
                peer_id=b_idn.key_id,
                identity_pub_hex=b_idn.public_bytes.hex(),
                fingerprint=b_idn.key_id,
                consent_state="paired",
                relay_url=str(http.base_url),
                relay_mailbox=b_mbx,
                now=now,
            )
            b_store.upsert_peer(
                peer_id=a_idn.key_id,
                identity_pub_hex=a_idn.public_bytes.hex(),
                fingerprint=a_idn.key_id,
                consent_state="paired",
                relay_url=str(http.base_url),
                relay_mailbox=a_mbx,
                now=now,
            )
            stage("pairing", True)
        except Exception as exc:  # noqa: BLE001 — self-test must never raise
            stage("pairing", False, str(exc))
            return result(False)

        # 4. handshake (3-leg): A opens → push leg1 → B polls (auto-pushes leg2) → A polls (completes)
        try:
            leg1 = open_session(a_store, a_idn, peer_id=b_idn.key_id, now=now)
            a_client.push(leg1)
            poll_and_ingest(b_store, b_idn, b_client, now=now)   # responder: receives leg1, pushes leg2
            poll_and_ingest(a_store, a_idn, a_client, now=now)   # initiator: receives leg2, completes
            sess = a_store.get_active_session(b_idn.key_id)
            if sess is None or a_store.get_session_key(b_idn.key_id, sess["session_id"]) is None:
                stage("handshake", False, "no session key established")
                return result(False)
            stage("handshake", True)
            session_id = sess["session_id"]
        except Exception as exc:  # noqa: BLE001
            stage("handshake", False, str(exc))
            return result(False)

        # 5. message_sent (A → B, fixed canned text, no LLM)
        try:
            sent = send_message(
                a_store, a_idn, a_client,
                peer_id=b_idn.key_id,
                session_id=session_id,
                payload=_CANNED,
                now=now,
            )
            stage("message_sent", bool(sent), "" if sent else "send returned False")
            if not sent:
                return result(False)
        except Exception as exc:  # noqa: BLE001
            stage("message_sent", False, str(exc))
            return result(False)

        # 6. message_received (B polls + decrypts → transcript carries the canned text)
        try:
            poll_and_ingest(b_store, b_idn, b_client, now=now)
            from brain.kindled_link.views import peer_transcript as _pt
            rows = _pt(b_store, a_idn.key_id)
            got = any(r.get("text") == _CANNED["text"] for r in rows)
            stage("message_received", got, "" if got else "canned message not found in transcript")
            if not got:
                return result(False)
        except Exception as exc:  # noqa: BLE001
            stage("message_received", False, str(exc))
            return result(False)

        return result(True)
    finally:
        # 7. cleanup — best-effort drain temp mailboxes; close stores; drop tmp dirs.
        with contextlib.suppress(Exception):
            if a_client is not None:
                fetched = a_client.fetch()
                if fetched:
                    a_client.ack([f["id"] for f in fetched])
        with contextlib.suppress(Exception):
            if b_client is not None:
                fetched = b_client.fetch()
                if fetched:
                    b_client.ack([f["id"] for f in fetched])
        for st in (a_store, b_store):
            with contextlib.suppress(Exception):
                if st is not None:
                    st.close()
        for d in (a_dir, b_dir):
            with contextlib.suppress(Exception):
                shutil.rmtree(d, ignore_errors=True)
        # cleanup stage always appended last, always ok (exceptions were suppressed above)
        stages.append({"name": "cleanup", "ok": True, "detail": ""})
