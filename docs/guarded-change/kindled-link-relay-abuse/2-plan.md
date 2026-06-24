# 2 — Plan

TDD, RED→GREEN. All relay changes live in `relay/dev_relay.py`; the one brain touch is
the `_payload_text` cap in `privacy_gate.py`. Constants get a small block at the top of
`dev_relay.py` (relay-local, env-overridable in `main()`), matching the existing
`_NONCE_TTL_SECONDS` convention.

## Constants (relay-local, top of dev_relay.py)

```
RELAY_MAX_ENVELOPE_BYTES  = 64 * 1024     # 64 KiB — generous for an opaque envelope
RELAY_MAX_QUEUE_DEPTH     = 256           # per-mailbox undelivered envelopes
RELAY_MAX_MAILBOXES       = 1024          # total distinct registered mailboxes
RELAY_RATE_MAX            = 60            # requests per window per client
RELAY_RATE_WINDOW_SECONDS = 60.0
```
Generous so the in-process test relay's normal flows are unaffected; the abuse cases
in the criteria push well past them (or use injected small caps via create_app kwargs).

## Changes

### dev_relay.py — `_Store`

- `push(envelope, *, max_bytes, max_depth, max_mailboxes)` → raise a typed rejection
  (`_RelayReject(status, detail)`) on oversize (413), new-mailbox-beyond-global-cap
  (429), or full-queue (`len >= max_depth` → 429); keep the pristine-envelope behaviour
  (no mutation). Size via `len(canonical_json(envelope).encode())` (stable — an attacker
  can't pad with whitespace; the recomputed size, not Content-Length, is the measure).
- `register(mailbox_id, pub, *, max_mailboxes)` → reject a NEW mailbox beyond the cap
  (over the union of registered + pushed-to keys); existing mailbox returns owner.
- `is_registered(mailbox_id)` helper (owner is not None) — for the challenge gate.
- `_mailbox_count()` = distinct keys across `mailboxes` ∪ `owners` (the cap denominator).

### dev_relay.py — endpoints

- `/envelope`: **stays OPEN (no registration gate — stage-3 blocker).** Order: 413 if
  oversize (cheapest, before any mailbox work) → 429 if this is a NEW mailbox key beyond
  `RELAY_MAX_MAILBOXES` → 429 if the target mailbox queue is full (`len >= max_depth`) →
  else push. (Wrap `_RelayReject` → `HTTPException`.)
- `/mailbox/challenge`: 404 if mailbox not registered (safe — owner registers first);
  else issue nonce as today.
- `/mailbox/register`: enforce `RELAY_MAX_MAILBOXES` over the union of registered +
  pushed-to keys; 429 when full + new id.
- Rate-limited endpoints (`/envelope`, `/mailbox/challenge`, `/mailbox/register`,
  fetch/ack): a `_RateLimiter` checked first; 429 on exceed.

### dev_relay.py — `_RateLimiter`

Fixed-window counter, clock-injected (reuse the `clock` arg threaded into
`create_app`/`_Store`). **Key by `mailbox_id` where the request carries one (challenge/
fetch/ack/register/envelope all do), else the client host** — so co-located brains
behind one NAT IP don't share a bucket (stage-3 major-risk). `allow(key, now)` → bool;
prune ALL stale windows opportunistically (not just the touched key) to bound the dict.
Fail-safe: unresolvable key → allow (C-9; documented as a LAN-only fail-open assumption).
Cap default generous (`RELAY_RATE_MAX=60`/`60s` per key) so one tick's ~5 requests + the
3× transport retry stay well under.

### relay_client.py — fail-soft on 4xx/5xx (C-12, stage-3 MAJOR — NEW brain scope)

`_post` currently returns the `Response` on any non-exception; callers do
`.json()["nonce"]`/`["envelopes"]` unchecked → a 404/413/429 body KeyErrors and aborts
the poll. Fix: after the `post`, if `response.status_code >= 400`, raise
`RelayUnavailableError(f"relay rejected {path}: {status}")` (treated by callers exactly
like a transport failure — the tick guard already swallows `RelayUnavailableError`). This
makes every new relay rejection fail-soft (the deposit/poll just doesn't complete this
tick; retried next tick). One-line guard; no caller signature change.

### privacy_gate.py — `_payload_text` cap (C-10)

Add `MAX_SCAN_CHARS` (e.g. 8192) in `privacy_gate.py`; in `_payload_text`, cap the BODY
to `MAX_SCAN_CHARS` FIRST, then append the relationship-hint JSON (or the fail-closed
sentinel) — so the sentinel that forces a prefilter hold always survives truncation
(stage-3 minor). Truncation bounds only the deterministic `_prefilter` scan; the LLM
reflection still receives the full untruncated `payload.body` via `_build_gate_prompt`
(verified), and the gate fails closed to hold on uncertainty — so the trade only shortens
the cheap backstop, never the load-bearing gate. Document the trade.

## Measurement / instrumentation

Each criterion is a direct `TestClient` status-code / state assertion or a wall-time
bound (C-10). No new logging needed; HTTP status + `store` state are the observable
signals. The durable-storage seam (C-11) is asserted structurally (the `_Store` public
surface), not behaviorally.

## Thresholds (finding → routing)

- A change that lets the relay MUTATE an envelope, or inspect plaintext → **blocker**.
- A cap that breaks an existing happy-path test (C-2/C-4 regression) → **major**.
- A rate-limiter that can 500 (not fail-safe) → **major** (C-9).
- Missing one abuse vector with no rationale → **minor**.
- Naming / status-code choice (413 vs 507) nits → **nitpick**.

## Gating vs advisory

- **Gating:** C-1…C-11, R1 (suite green), R2 (ruff).
- **Advisory:** R3 (full backend).
- **N/A:** Layer-2 cost/cache (off chat path).

## Deferred (recorded, NOT in this change)

- **Real durable storage backend** (disc/sqlite mailboxes surviving restart). This
  change ships the in-memory `_Store` behind a stable seam (C-11); a durable backend is
  an infra deliverable (deployment, persistence schema, fsync discipline) — kept out of
  landable code.
- **Live cross-machine validation (#51)** — needs ≥2 machines + a deployed relay.
- **Tighter email regex** — the `_payload_text` cap bounds the ReDoS; re-anchoring
  `[\w.+-]+@...` to remove the backtracking is a separate small follow-up.
- **Per-mailbox push auth / sender allowlist** — `/envelope` stays open by protocol
  design (senders can't prove ownership; store-and-forward must accept deposits to
  not-yet-registered recipient mailboxes); caps + rate-limit are the bound. A sender-side
  proof is a protocol change, out of scope.
- **Fetch/ack response egress** — a mailbox at `RELAY_MAX_QUEUE_DEPTH` × envelope-size is
  returned whole in one fetch JSON (the brain processes only `INBOUND_FLOOD_CAP=20` but
  the relay still ships all). Bounding the fetch response (paginate / cap returned count)
  is a follow-up; named here as a known residual (stage-3 minor). Queue-depth + size caps
  bound the absolute worst case meanwhile.

---

## Revision 2 — full corrected design (gate-4 human tie-break, Hana)

Supersedes the single-global-cap + single-rate-key + relay-client-only fail-soft.

### dev_relay.py — `_Store` split pools

- Track registration status: `owners` = registered mailboxes; an `unregistered: dict`
  (insertion-ordered) for mailbox keys created by push that were never registered. A
  mailbox key lives in exactly one pool; `register()` on an unregistered key MOVES it
  (delete from `unregistered`, add to `owners`) — frees an unregistered slot (C-7b).
- Constants: `RELAY_MAX_REGISTERED_MAILBOXES = 1024`,
  `RELAY_MAX_UNREGISTERED_MAILBOXES = 256`.
- `push`: if target is a known key (registered or unregistered) → size+queue checks only.
  If NEW: it's unregistered; if the unregistered pool is full, **evict the oldest
  unregistered key whose queue is empty** (iterate insertion order, drop the first empty
  one); if NONE empty → 429 (never drop deposited mail). Then add the new key (C-13).
- `register`: NEW registered key beyond `RELAY_MAX_REGISTERED_MAILBOXES` → 429; existing
  → owner; unregistered→registered MOVE always allowed (frees a slot, C-7b/C-14).
- Order in `/envelope`: 413 size → (new-key eviction/429) → queue-depth 429 → push.

### dev_relay.py — `_RateLimiter` split keying

- `/envelope`: key = client host (sender) — `request.client.host`, None-safe (C-8b/C-9).
- `/mailbox/{challenge,fetch,ack,register}`: key = `mailbox_id` (caller's own identity).
- Fixed-window, clock-injected, prune ALL stale windows on check.

### brain — poll-path fail-soft (C-12 + C-12b + C-15)

- `relay_client._post`: after post, `if resp.status_code >= 400: raise
  RelayUnavailableError(f"relay {path} -> {status}")`. (4xx is NOT a TransportError so it
  does not re-enter the 3× retry loop — verified.)
- `transport.poll_and_ingest`: wrap the `relay.fetch()` call in try/except
  `RelayUnavailableError` → return a degraded/empty result (no raise), so
  `run_kindled_link_tick` completes + saves cadence (C-12b).
- **Self-heal (C-15):** make the tick's poll path call `relay.register()` idempotently
  before fetch (register is first-write-wins / cheap), so a relay restart that lost the
  registration re-establishes it before the challenge — no permanent 404 wedge. (Confirm
  during build where register() is currently called; if the tick already registers each
  cycle, C-15 is satisfied and only needs a test; else add the idempotent call.)

### Constants summary (rev2)

```
RELAY_MAX_ENVELOPE_BYTES        = 64 * 1024
RELAY_MAX_QUEUE_DEPTH           = 256
RELAY_MAX_REGISTERED_MAILBOXES  = 1024
RELAY_MAX_UNREGISTERED_MAILBOXES= 256
RELAY_RATE_MAX                  = 60
RELAY_RATE_WINDOW_SECONDS       = 60.0
MAX_SCAN_CHARS                  = 8192   # privacy_gate
```

### Eviction safety invariant (the load-bearing one)
Eviction NEVER drops an envelope: only EMPTY unregistered mailboxes are evicted; a full
pool of non-empty unregistered mailboxes 429s the new push instead. Registered mailboxes
are never push-evicted. This keeps the store-and-forward guarantee the stage-3 blocker
was about.

### Deferred (added at gate-7) — residual targeted queue-fill DoS
With `/envelope` open + per-mailbox `RELAY_MAX_QUEUE_DEPTH=256`, an attacker (capped to
their own host rate budget) can slowly fill a SPECIFIC victim's inbound queue over ~minutes,
429-ing NEW senders to that victim until the victim acks/drains. Bounded (256 cap), self-
healing (ack frees slots), host-rate-limited (not instant), and scoped to the trusted-LAN
posture. NOT the cross-tenant rate DoS (that's closed) — the victim can still read their own
mail. **Phase-7b public-relay checklist item:** sender-authenticated push and/or per-sender
queue quotas before any public relay. Recorded, accepted for the trusted-LAN/off-by-default
posture.
