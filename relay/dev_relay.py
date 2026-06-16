"""Minimal dumb relay (protocol §8 / design §8). Stores opaque encrypted
envelopes by mailbox and forwards them; never sees plaintext. In-memory state
for the dev/alpha relay."""
from __future__ import annotations

import itertools
import secrets

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import verify as _verify


class _Store:
    def __init__(self) -> None:
        self.mailboxes: dict[str, list[dict]] = {}
        self._ids = itertools.count(1)
        self.owners: dict[str, str] = {}   # mailbox_id → identity_pub hex
        self.nonces: dict[str, set[str]] = {}  # mailbox_id → live nonces

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
    if nonce not in store.nonces.get(mailbox_id, set()):
        raise HTTPException(status_code=401, detail="bad nonce")
    body = canonical_json({"purpose": "kindled-relay-auth/1",
                           "mailbox": mailbox_id, "nonce": nonce})
    try:
        ok = _verify(bytes.fromhex(identity_pub), bytes.fromhex(signature), body)
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(status_code=401, detail="bad signature")
    store.nonces[mailbox_id].discard(nonce)  # single-use


def create_app(require_auth: bool = False) -> FastAPI:
    app = FastAPI()
    store = _Store()

    @app.post("/mailbox/register")
    def register(req: _RegisterReq) -> dict:
        store.owners.setdefault(req.mailbox_id, req.identity_pub)  # first-write-wins
        return {"ok": True, "owner": store.owners[req.mailbox_id]}

    @app.post("/mailbox/challenge")
    def challenge(req: _FetchReq) -> dict:
        nonce = secrets.token_hex(16)
        store.nonces.setdefault(req.mailbox_id, set()).add(nonce)
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
