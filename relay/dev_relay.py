"""Minimal dumb relay (protocol §8 / design §8). Stores opaque encrypted
envelopes by mailbox and forwards them; never sees plaintext. In-memory state
for the dev/alpha relay. Mailbox-ownership auth on fetch/ack is added in Task 8."""
from __future__ import annotations

import itertools

from fastapi import FastAPI
from pydantic import BaseModel


class _Store:
    def __init__(self) -> None:
        self.mailboxes: dict[str, list[dict]] = {}
        self._ids = itertools.count(1)

    def push(self, envelope: dict) -> str:
        mbx = envelope["relay_mailbox"]
        env_id = f"env_{next(self._ids)}"
        self.mailboxes.setdefault(mbx, []).append({"id": env_id, **envelope})
        return env_id

    def fetch(self, mailbox_id: str) -> list[dict]:
        return list(self.mailboxes.get(mailbox_id, []))

    def ack(self, mailbox_id: str, env_ids: list[str]) -> None:
        kept = [e for e in self.mailboxes.get(mailbox_id, []) if e["id"] not in set(env_ids)]
        self.mailboxes[mailbox_id] = kept


class _FetchReq(BaseModel):
    mailbox_id: str


class _AckReq(BaseModel):
    mailbox_id: str
    envelope_ids: list[str]


def create_app() -> FastAPI:
    app = FastAPI()
    store = _Store()

    @app.post("/envelope")
    def post_envelope(envelope: dict) -> dict:  # open push, rate-limited in prod
        return {"id": store.push(envelope)}

    @app.post("/mailbox/fetch")
    def fetch(req: _FetchReq) -> dict:
        return {"envelopes": store.fetch(req.mailbox_id)}

    @app.post("/mailbox/ack")
    def ack(req: _AckReq) -> dict:
        store.ack(req.mailbox_id, req.envelope_ids)
        return {"ok": True}

    app.state.store = store
    return app
