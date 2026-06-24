"""Minimal dumb relay (protocol §8 / design §8). Stores opaque encrypted
envelopes by mailbox and forwards them; never sees plaintext. In-memory state
for the dev/alpha relay.

#48 abuse/quota hardening (guarded-change kindled-link-relay-abuse): envelope
size cap, per-mailbox queue depth, split registered/unregistered mailbox pools
with empty-first eviction (an open-push flood can't lock out honest onboarding),
host-keyed rate limiting (every endpoint by client host — never the
attacker-suppliable mailbox_id), and a challenge registration gate. Crypto/opacity unchanged: the
relay still never inspects plaintext and never mutates an envelope.
"""
from __future__ import annotations

import itertools
import secrets
import time
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from brain.kindled_link.codec import canonical_json
from brain.kindled_link.identity import verify as _verify

# Challenge nonces expire after this many seconds (#49).
_NONCE_TTL_SECONDS = 120.0

# #48 abuse/quota caps (env-overridable in main()). Generous so the in-process
# test relay's normal flows are unaffected; abuse cases push well past them.
RELAY_MAX_ENVELOPE_BYTES = 64 * 1024          # 64 KiB per opaque envelope
RELAY_MAX_QUEUE_DEPTH = 256                    # undelivered envelopes per mailbox
RELAY_MAX_REGISTERED_MAILBOXES = 1024          # honest, register-gated pool
RELAY_MAX_UNREGISTERED_MAILBOXES = 256         # pushed-to-but-never-registered pool
RELAY_RATE_MAX = 60                            # requests per window per key
RELAY_RATE_WINDOW_SECONDS = 60.0


class _RelayRejectError(Exception):
    """A bounded-resource rejection; carries the HTTP status to surface."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _RateLimiter:
    """Fixed-window per-key counter, clock-injected. Fail-safe: a None key is
    always allowed (an unresolvable client must never 500 the relay). Prunes all
    stale windows on each check so the keyed dict can't accrete dead keys."""

    def __init__(self, max_requests: int, window: float) -> None:
        self._max = max_requests
        self._window = window
        self._buckets: dict[str, list[float]] = {}  # key -> [window_start, count]

    def allow(self, key: str | None, now: float) -> bool:
        if key is None:
            return True
        # prune stale windows (bounds the dict against one-shot keys)
        stale = [k for k, (start, _) in self._buckets.items()
                 if now - start >= self._window]
        for k in stale:
            del self._buckets[k]
        bucket = self._buckets.get(key)
        if bucket is None or now - bucket[0] >= self._window:
            self._buckets[key] = [now, 1]
            return self._max >= 1
        bucket[1] += 1
        return bucket[1] <= self._max


class _Store:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.mailboxes: dict[str, list[dict]] = {}
        self._ids = itertools.count(1)
        self.owners: dict[str, str] = {}            # registered mailbox -> identity_pub hex
        self.unregistered: dict[str, None] = {}     # insertion-ordered pushed-to keys
        self._clock = clock
        self.nonces: dict[str, dict[str, float]] = {}

    # --- nonce machinery (#49) ---
    def _prune_nonces(self, mailbox_id: str) -> None:
        live = self.nonces.get(mailbox_id)
        if not live:
            return
        now = self._clock()
        for n in [n for n, exp in live.items() if exp <= now]:
            del live[n]
        if not live:
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

    # --- mailbox pools (#48) ---
    def is_registered(self, mailbox_id: str) -> bool:
        return mailbox_id in self.owners

    def _is_known(self, mailbox_id: str) -> bool:
        return mailbox_id in self.owners or mailbox_id in self.unregistered

    def register(self, mailbox_id: str, identity_pub: str, *,
                 max_registered: int = RELAY_MAX_REGISTERED_MAILBOXES) -> str:
        if mailbox_id in self.owners:
            return self.owners[mailbox_id]            # first-write-wins
        if len(self.owners) >= max_registered:
            raise _RelayRejectError(429, "registered mailbox cap reached")
        # promote an unregistered pushed-to key (frees an unregistered slot)
        self.unregistered.pop(mailbox_id, None)
        self.owners[mailbox_id] = identity_pub
        return identity_pub

    def _evict_oldest_empty_unregistered(self) -> bool:
        """Evict the oldest unregistered mailbox whose queue is EMPTY (never one
        holding undelivered envelopes — that would drop deposited mail). Returns
        True if a slot was freed."""
        for mid in list(self.unregistered):           # insertion order
            if not self.mailboxes.get(mid):
                del self.unregistered[mid]
                self.mailboxes.pop(mid, None)
                self.nonces.pop(mid, None)
                return True
        return False

    def push(self, envelope: dict, *,
             max_bytes: int = RELAY_MAX_ENVELOPE_BYTES,
             max_depth: int = RELAY_MAX_QUEUE_DEPTH,
             max_unregistered: int = RELAY_MAX_UNREGISTERED_MAILBOXES) -> str:
        if len(canonical_json(envelope)) > max_bytes:
            raise _RelayRejectError(413, "envelope too large")
        mbx = envelope["relay_mailbox"]
        if not self._is_known(mbx):
            # a NEW key lands in the unregistered pool; make room without ever
            # dropping deposited mail.
            if len(self.unregistered) >= max_unregistered \
                    and not self._evict_oldest_empty_unregistered():
                raise _RelayRejectError(429, "unregistered mailbox pool full")
            self.unregistered[mbx] = None
        if len(self.mailboxes.get(mbx, [])) >= max_depth:
            raise _RelayRejectError(429, "mailbox queue full")
        env_id = f"env_{next(self._ids)}"
        # tracking id kept SEPARATE from the opaque envelope — never merged in
        # (mutating it would change canonical JSON + break the signature).
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
    require_auth: bool = False,
    clock: Callable[[], float] = time.monotonic,
    *,
    max_envelope_bytes: int = RELAY_MAX_ENVELOPE_BYTES,
    max_queue_depth: int = RELAY_MAX_QUEUE_DEPTH,
    max_registered_mailboxes: int = RELAY_MAX_REGISTERED_MAILBOXES,
    max_unregistered_mailboxes: int = RELAY_MAX_UNREGISTERED_MAILBOXES,
    rate_max: int = RELAY_RATE_MAX,
    rate_window: float = RELAY_RATE_WINDOW_SECONDS,
) -> FastAPI:
    app = FastAPI()
    store = _Store(clock=clock)
    limiter = _RateLimiter(rate_max, rate_window)

    def _rate(request: Request) -> None:
        # Rate-limit by CLIENT HOST (the requester), uniformly on every endpoint.
        # NOT by mailbox_id: an attacker can put a VICTIM's mailbox_id in an
        # unauthenticated /mailbox/fetch|ack|challenge body, and since the rate
        # check runs before auth, a mailbox-keyed limiter would let the attacker
        # drain the victim's budget (cross-tenant DoS — stage-6 finding). Keying
        # on the requester's own host confines each actor to its own budget.
        # Trade-off: co-located honest brains behind one NAT share a host bucket;
        # acceptable for the LAN/trusted posture given the generous default cap.
        key = request.client.host if request.client else None  # fail-open if None
        if not limiter.allow(key, clock()):
            raise HTTPException(status_code=429, detail="rate limit exceeded")

    @app.post("/mailbox/register")
    def register(req: _RegisterReq, request: Request) -> dict:
        _rate(request)
        try:
            owner = store.register(req.mailbox_id, req.identity_pub,
                                   max_registered=max_registered_mailboxes)
        except _RelayRejectError as e:
            raise HTTPException(status_code=e.status, detail=e.detail) from None
        return {"ok": True, "owner": owner}

    @app.post("/mailbox/challenge")
    def challenge(req: _FetchReq, request: Request) -> dict:
        _rate(request)
        # registration gate (#48): only a registered mailbox gets a nonce, so an
        # unauthenticated flood of distinct ids can't accrete the nonce dict.
        if not store.is_registered(req.mailbox_id):
            raise HTTPException(status_code=404, detail="mailbox not registered")
        nonce = secrets.token_hex(16)
        store.issue_nonce(req.mailbox_id, nonce)
        return {"nonce": nonce}

    @app.post("/envelope")
    def post_envelope(envelope: dict, request: Request) -> dict:
        # /envelope stays OPEN (store-and-forward must accept deposits to
        # recipient mailboxes not yet registered). Caps + split-pool eviction
        # bound memory; the host-keyed rate limit bounds push flooding.
        _rate(request)
        try:
            env_id = store.push(envelope, max_bytes=max_envelope_bytes,
                                max_depth=max_queue_depth,
                                max_unregistered=max_unregistered_mailboxes)
        except _RelayRejectError as e:
            raise HTTPException(status_code=e.status, detail=e.detail) from None
        return {"id": env_id}

    @app.post("/mailbox/fetch")
    def fetch(req: _FetchReq, request: Request) -> dict:
        _rate(request)
        if require_auth:
            _check_auth(store, req.mailbox_id, req.nonce, req.signature, req.identity_pub)
        return {"envelopes": store.fetch(req.mailbox_id)}

    @app.post("/mailbox/ack")
    def ack(req: _AckReq, request: Request) -> dict:
        _rate(request)
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
    127.0.0.1 or a trusted LAN only — NOT a public interface. The #48 abuse caps
    (size/queue/split-pool/rate-limit) bound memory + onboarding-lockout, but a
    PUBLIC relay still needs durable storage + the deployment hardening in the
    public-relay checklist (Phase 7b). To run:
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
