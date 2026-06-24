# 1 — Spec: Kindled relay abuse / quota hardening (#48)

## Problem

`relay/dev_relay.py` is the dumb mailbox relay (protocol §8). It is correct on the
crypto axis (ciphertext-opaque, never sees plaintext, mailbox-ownership auth on
fetch/ack) but has **no abuse controls** — flagged repeatedly: the #49 fix bounded the
inner nonce set per mailbox but explicitly left the OUTER vectors to #48
(`dev_relay.py:17-24`), and `main()`'s own docstring says "bind to 127.0.0.1 / trusted
LAN only … abuse hardening (rate limits, durable storage, quotas) is Phase 7b"
(`dev_relay.py:165-167`). The relay cannot move toward a public/multi-machine posture
until these land.

### Abuse vectors (all in `relay/dev_relay.py`)

1. **Open unbounded push.** `/envelope` (`:137-139`) is unauthenticated by design (a
   sender can't prove mailbox ownership). It accepts ANY payload of ANY size to ANY
   `relay_mailbox`, appended without limit (`_Store.push`, `:62-70`). A flooder
   exhausts relay memory and floods the recipient's poll (the brain bounds its own
   per-poll work at `INBOUND_FLOOD_CAP=20`, `transport.py:135`, but the relay-side
   queue still grows unbounded and never drains for excess).
2. **Push to arbitrary / unregistered mailboxes.** `push` keys on
   `envelope["relay_mailbox"]` with no check that the mailbox was ever registered →
   unbounded distinct-mailbox keyspace in `_Store.mailboxes`.
3. **Unauthenticated challenge / register accretion.** `/mailbox/challenge`
   (`:131-135`) and `/mailbox/register` (`:126-129`) accept any `mailbox_id`. The #49
   fix bounds the per-mailbox nonce set but NOT the count of distinct mailbox keys in
   `_Store.nonces` / `_Store.owners` — a flood of distinct `mailbox_id`s accretes
   those outer dicts (the residual #49 explicitly named).
4. **No envelope size cap.** A single multi-MB envelope is accepted whole.
5. **No request rate limiting.** No per-client throttle on any endpoint.
6. **In-memory only.** State is lost on restart (acceptable for dev; a real relay needs
   durable storage — this part is infra, see Deferred).

### Bundled brain-side finding (from the m9/m10 stage-6 red-team)

7. **Pre-existing ReDoS in the privacy-gate email pattern.**
   `_PATTERNS` email regex `[\w.+-]+@[\w-]+\.[\w.-]+` (`privacy_gate.py:30`) backtracks
   catastrophically on a long no-`@` input (~34s on `"eyJ"+"A"*100000`). `_payload_text`
   scans the Kindled's own outbound draft body (bounded by her generation length →
   low reachability, but real). A length cap on the scanned text bounds ALL pre-filter
   regex work cheaply. Folded here as defence-in-depth abuse hardening (same theme).

## Goals

Make the dev/alpha relay safe to run on a trusted LAN / toward a public posture by
bounding every unbounded resource, WITHOUT changing the crypto/opacity guarantees or
the brain↔relay protocol. Specifically:

- Cap envelope size; reject oversize (HTTP 413).
- Bound per-mailbox queue depth; reject when full (HTTP 429).
- **`/envelope` stays OPEN** (store-and-forward must accept deposits to recipient
  mailboxes that self-register only on their own later tick — see stage-3 blocker).
  Bound the mailbox keyspace by a **global distinct-mailbox cap on first touch** (push
  or register), not by requiring pre-registration.
- Gate `/mailbox/challenge` on registration (an owner registers before its own first
  challenge — safe; closes the #49 nonce outer-accretion residual).
- Cap the total number of distinct mailbox keys (bounds `_Store.owners`/`mailboxes`).
- Per-client (IP) rate limit on `/envelope`, `/mailbox/challenge`, `/mailbox/register`
  via an in-memory token bucket (clock-injected for tests), fail-safe.
- Cap `_payload_text` length in `privacy_gate.py` (bundled ReDoS bound).

## Constraints

- **No crypto/opacity change.** The relay still never inspects plaintext; `push` must
  not mutate the envelope (the pristine-envelope invariant, `dev_relay.py:64-69`).
- **Minimal brain change.** Request SHAPES are unchanged. But the new relay rejections
  (413/429/404) do NOT fail-soft today (stage-3 MAJOR): `relay_client._post` returns the
  Response unchecked and callers `.json()["nonce"]`/`["envelopes"]` KeyError-crash on a
  4xx body. So the brain touches are: (a) `relay_client._post` raises
  `RelayUnavailableError` on 4xx/5xx (one guard), and (b) the `_payload_text` cap. Both
  are additive + fail-soft; no signature changes.
- **Backwards-compatible defaults.** Existing tests construct `create_app(...)`; new
  caps must default to values that don't break the in-process test relay's normal flows
  (small envelopes, few mailboxes). Caps are generous + override-able.
- **Durable storage is a pluggable-interface DESIGN here, in-memory default.** A real
  durable backend (disc/db) + live multi-machine validation (#51) is infra — out of
  scope for landable code; the spec records the interface seam.

## Prior art

- #49 nonce TTL + inner-set bound: `dev_relay.py:17-60`, `test_dev_relay.py`.
- Brain-side inbound flood cap: `INBOUND_FLOOD_CAP`, `transport.py:135`.
- Ledger: kindled #48 (`project_companion_emergence_kindled_link.md`,
  `project_companion_emergence_deferred.md` Phase-2 §48).
