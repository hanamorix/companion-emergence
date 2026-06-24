# 3 — Plan red-team (two cold laps)

## Lap 1 — verdict BLOCKER
- **BLOCKER (C-5):** registration-gating `/envelope` breaks store-and-forward — recipients
  self-register lazily on their own tick; the in-memory relay loses registrations on restart;
  the existing `test_push_then_fetch` pushes to an unregistered mailbox. Gating drops deposits
  incl. handshake leg1. → DROP the gate; bound keyspace by a global first-touch cap.
- **MAJOR:** new 4xx codes KeyError-crash the brain (`relay_client._post` catches only
  `TransportError`; callers read `.json()["nonce"]` unchecked). → add `relay_client` fail-soft.
- Minors: cap body-then-hint (sentinel), rate-key mailbox-else-host, 413-first, queue `>=`,
  fetch-egress residual. Citations verified (relay_client.py:30/44/52, tick.py, test_dev_relay).

## Lap 2 — verdict MAJOR ×3 (after the global-cap revision)
- **MAJOR (keyspace lockout):** open /envelope + single global cap → attacker fills the cap
  with garbage keys → honest onboarding denied. → SPLIT pools (registered/unregistered) +
  empty-first LRU eviction that never drops mail.
- **MAJOR (cross-tenant push DoS):** rate-limiter keyed on mailbox_id = recipient on open
  /envelope → attacker drains a victim's budget. → key /envelope on sender host.
- **MAJOR (C-12 incomplete):** `poll_and_ingest` fetch unguarded (tick.py:83) → 4xx aborts the
  tick. → guard the poll + idempotent re-register.
- C-10 ReDoS body-cap + payload.body/sentinel claims clean.

## Gate 4 routing
Lap1 BLOCKER + Lap2 MAJOR×3 (2nd gate-4 bounce → iteration cap → human tie-break). Hana chose
the FULL corrected design (Revision 2 in 1.5-criteria/2-plan): split pools + empty-first LRU +
sender-keyed limiter + poll guard + supervisor re-register. Proceeded to build; stage 6 verifies.
