# 6 — Code red-team (two cold laps)

## Lap 1 — verdict MAJOR ×2
Headline blocker-fixes verified CLEAN: split-pool `_evict_oldest_empty_unregistered` only
evicts EMPTY unregistered mailboxes (never drops deposited mail); `register` moves
unregistered→registered + frees the slot; `/envelope` keys the limiter on `request.client.host`;
`_post` does not retry a 4xx; `_payload_text` caps body-then-hint (LLM sees full body). Two MAJORs:
- **rate-before-auth on /mailbox/fetch|ack** keyed on the attacker-suppliable victim mailbox_id
  → cross-tenant poll-budget DoS (same class C-8b closed for /envelope).
- **C-15** marked gating but (reviewer grepped tick.py only) — RESOLVED as a reviewer miss:
  supervisor.py:664 calls `relay.register()` every tick (guarded) before the poll.

## Fix (build)
- Rate-limit ALL endpoints by **client host** uniformly (supersedes Rev-2 mailbox-id keying for
  mailbox endpoints). Closes the cross-tenant DoS at the source (host is the TCP peer, not
  body-suppliable). Regression pin `test_c8b_fetch_rate_keyed_by_host_not_mailbox_id`.
- C-15 relay-side self-heal pin `test_c15_reregister_after_relay_restart_restores_challenge`;
  brain side accepted as supervisor.py:664 (pre-existing, guarded).

## Lap 2 (re-review of the fix) — verdict MAJOR (non-blocking, accepted)
- Cross-tenant rate DoS **CLOSED, not relocated** (host derived from TCP peer; unforgeable).
- Split-pool eviction still correct; normal tick ~6 req << 60/60s cap.
- **Residual MAJOR — ACCEPTED/DEFERRED:** open /envelope + 256 queue-depth → an attacker
  (capped to own host budget) can slowly fill a SPECIFIC victim's queue, 429-ing new senders
  (victim still reads own mail). Bounded, self-healing on ack, host-rate-limited, trusted-LAN
  scoped → Phase-7b public-relay checklist (sender-auth / per-sender queue quota).
- Minors fixed: stale module/test docstrings ("mailbox endpoints by mailbox_id") corrected.

## Gate 7 routing
MAJOR×2 lap1 → fixed in build → re-review → residual MAJOR accepted as a documented Phase-7b
deferral; minors fixed. Proceed to stage 8.
