"""Minimal dumb relay (protocol §8 / design §8). Stores opaque encrypted
envelopes by mailbox and forwards them; never sees plaintext. In-memory state
for the dev/alpha relay."""
from __future__ import annotations

import itertools
import secrets
import time
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import verify as _verify

# Challenge nonces expire after this many seconds (#49). Short window: an unused
# or stolen nonce can't replay indefinitely, and a mailbox's live-nonce set can't
# grow unbounded (expired entries are pruned on every issue/check, and an emptied
# mailbox entry is dropped). NOTE: the OUTER count of distinct mailbox keys is NOT
# bounded here — `/mailbox/challenge` is unauthenticated, so a flood of distinct
# mailbox_ids is an abuse vector left to 7b (durable relay + rate-limits/quotas,
# defer #48). A prod relay may tune the TTL; the dev/alpha relay enforces one so
# the dev posture isn't laxer.
_NONCE_TTL_SECONDS = 120.0


class _Store:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.mailboxes: dict[str, list[dict]] = {}
        self._ids = itertools.count(1)
        self.owners: dict[str, str] = {}   # mailbox_id → identity_pub hex
        self._clock = clock
        self.nonces: dict[str, dict[str, float]] = {}  # mailbox_id → {nonce: expires_at}

    def _prune_nonces(self, mailbox_id: str) -> None:
        live = self.nonces.get(mailbox_id)
        if not live:
            return
        now = self._clock()
        expired = [n for n, exp in live.items() if exp <= now]
        for n in expired:
            del live[n]
        if not live:  # drop the emptied mailbox entry so the outer dict doesn't accrete
            self.nonces.pop(mailbox_id, None)

    def issue_nonce(self, mailbox_id: str, nonce: str) -> None:
        self._prune_nonces(mailbox_id)
        self.nonces.setdefault(mailbox_id, {})[nonce] = self._clock() + _NONCE_TTL_SECONDS

    def nonce_live(self, mailbox_id: str, nonce: str) -> bool:
        self._prune_nonces(mailbox_id)
        return nonce in self.nonces.get(mailbox_id, {})

    def discard_nonce(self, mailbox_id: str, nonce: str) -> None:
        live = self.nonces.get(mailbox_id)
        if live is not None:
            live.pop(nonce, None)
            if not live:
                self.nonces.pop(mailbox_id, None)

    def push(self, envelope: dict) -> str:
        mbx = envelope["relay_mailbox"]
        env_id = f"env_{next(self._ids)}"
        # The relay's tracking id is transport metadata kept SEPARATE from the
        # opaque envelope — never merged into it. Injecting a field into the
        # envelope would change its canonical JSON and break the recipient's
        # signature verification (the relay must not mutate the payload).
        self.mailboxes.setdefault(mbx, []).append({"id": env_id, "envelope": envelope})
        return env_id

    def fetch(self, mailbox_id: str) -> list[dict]:
        return list(self.mailboxes.get(mailbox_id, []))

    def ack(self, mailbox_id: str, env_ids: list[str]) -> None:
        kept = [e for e in self.mailboxes.get(mailbox_id, []) if e["id"] not in set(env_ids)]
        self.mailboxes[mailbox_id] = kept


class _RegisterReq(BaseModel):
    mailbox_id: str
    identity_pub: str


class _FetchReq(BaseModel):
    mailbox_id: str
    nonce: str | None = None
    signature: str | None = None
    identity_pub: str | None = None


class _AckReq(BaseModel):
    mailbox_id: str
    envelope_ids: list[str]
    nonce: str | None = None
    signature: str | None = None
    identity_pub: str | None = None


def _check_auth(store: _Store, mailbox_id: str, nonce: str | None,
                signature: str | None, identity_pub: str | None) -> None:
    owner = store.owners.get(mailbox_id)
    if owner is None or nonce is None or signature is None or identity_pub is None:
        raise HTTPException(status_code=401, detail="auth required")
    if identity_pub != owner:
        raise HTTPException(status_code=401, detail="not mailbox owner")
    if not store.nonce_live(mailbox_id, nonce):
        raise HTTPException(status_code=401, detail="bad nonce")
    body = canonical_json({"purpose": "kindled-relay-auth/1",
                           "mailbox": mailbox_id, "nonce": nonce})
    try:
        ok = _verify(bytes.fromhex(identity_pub), bytes.fromhex(signature), body)
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(status_code=401, detail="bad signature")
    store.discard_nonce(mailbox_id, nonce)  # single-use


def create_app(
    require_auth: bool = False, clock: Callable[[], float] = time.monotonic
) -> FastAPI:
    app = FastAPI()
    store = _Store(clock=clock)

    @app.post("/mailbox/register")
    def register(req: _RegisterReq) -> dict:
        store.owners.setdefault(req.mailbox_id, req.identity_pub)  # first-write-wins
        return {"ok": True, "owner": store.owners[req.mailbox_id]}

    @app.post("/mailbox/challenge")
    def challenge(req: _FetchReq) -> dict:
        nonce = secrets.token_hex(16)
        store.issue_nonce(req.mailbox_id, nonce)
        return {"nonce": nonce}

    @app.post("/envelope")
    def post_envelope(envelope: dict) -> dict:  # open push, rate-limited in prod
        return {"id": store.push(envelope)}

    @app.post("/mailbox/fetch")
    def fetch(req: _FetchReq) -> dict:
        if require_auth:
            _check_auth(store, req.mailbox_id, req.nonce, req.signature, req.identity_pub)
        return {"envelopes": store.fetch(req.mailbox_id)}

    @app.post("/mailbox/ack")
    def ack(req: _AckReq) -> dict:
        if require_auth:
            _check_auth(store, req.mailbox_id, req.nonce, req.signature, req.identity_pub)
        store.ack(req.mailbox_id, req.envelope_ids)
        return {"ok": True}

    app.state.store = store
    return app


def main() -> None:
    """Run the self-hosted alpha relay (Phase 7a).

    Config via env: KINDLED_RELAY_HOST (default 127.0.0.1), KINDLED_RELAY_PORT
    (default 8787), KINDLED_RELAY_AUTH ("1"/"0", default "1" — fetch/ack require
    mailbox-ownership proof; `/envelope` push stays open by design).

    BIND CONSTRAINT (Phase 7a): `/envelope` is unauthenticated, so bind to
    127.0.0.1 or a trusted LAN only — NOT a public interface. Public-relay abuse
    hardening (rate limits, durable storage, quotas) is Phase 7b. To run:
        uv run python -m relay.dev_relay
    """
    import os

    import uvicorn

    host = os.environ.get("KINDLED_RELAY_HOST", "127.0.0.1")
    port = int(os.environ.get("KINDLED_RELAY_PORT", "8787"))
    require_auth = os.environ.get("KINDLED_RELAY_AUTH", "1") != "0"
    uvicorn.run(create_app(require_auth=require_auth), host=host, port=port)


if __name__ == "__main__":
    main()
