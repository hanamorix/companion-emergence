"""Outbound-only relay client (design §3, §16). Pure httpx caller: register,
push, challenge+sign, fetch, ack. NO inbound listener, NO server. Retries are
bounded with backoff; the caller decides cadence."""
from __future__ import annotations

import logging
import time

import httpx

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import KindledIdentity

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_S = 0.5


class RelayClient:
    def __init__(self, http: httpx.Client, *, identity: KindledIdentity, mailbox_id: str) -> None:
        self._http = http
        self._idn = identity
        self._mailbox = mailbox_id

    def _post(self, path: str, payload: dict) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._http.post(path, json=payload)
            except httpx.TransportError as exc:  # network errors only — bounded backoff
                last_exc = exc
                time.sleep(_BACKOFF_BASE_S * (2**attempt))
        raise RelayUnavailableError(str(last_exc))

    def register(self) -> None:
        self._post("/mailbox/register",
                   {"mailbox_id": self._mailbox, "identity_pub": self._idn.public_bytes.hex()})

    def push(self, envelope: dict) -> None:
        self._post("/envelope", envelope)

    def _auth(self) -> dict:
        nonce = self._post("/mailbox/challenge", {"mailbox_id": self._mailbox}).json()["nonce"]
        body = canonical_json({"purpose": "kindled-relay-auth/1",
                               "mailbox": self._mailbox, "nonce": nonce})
        return {"mailbox_id": self._mailbox, "nonce": nonce,
                "signature": self._idn.sign(body).hex(),
                "identity_pub": self._idn.public_bytes.hex()}

    def fetch(self) -> list[dict]:
        return self._post("/mailbox/fetch", self._auth()).json()["envelopes"]

    def ack(self, envelope_ids: list[str]) -> None:
        self._post("/mailbox/ack", {**self._auth(), "envelope_ids": envelope_ids})


class RelayUnavailableError(RuntimeError):
    """The relay could not be reached after bounded retries."""
