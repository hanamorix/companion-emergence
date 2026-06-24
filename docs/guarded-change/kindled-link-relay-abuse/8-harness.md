# 8 — Harness (conformance + regression)

No stage-0 baseline (Layer-2 cost/cache metrics are off the relay/gate path — not a
comparable workload). Stage 8 = conformance (always) + the kindled_link suite as the
regression oracle.

## Conformance — vs 1.5-criteria (incl. Revision 2)

| Criterion | Result |
|---|---|
| C-1/C-2 envelope size cap | PASS (413 oversize not stored; normal pushes) |
| C-3/C-4 per-mailbox queue depth | PASS (429 at cap; ack frees capacity) |
| C-5b/C-13 split-pool + empty-first eviction | PASS (evicts oldest EMPTY; 429s when all non-empty, drops nothing) |
| C-7b/C-14 register cap + unregistered→registered move + no honest lockout | PASS |
| C-6 challenge registration gate | PASS (unregistered → 404/401) |
| C-8/C-8b host-keyed rate limit + cross-tenant closed | PASS (`test_c8b_fetch_rate_keyed_by_host_not_mailbox_id`) |
| C-9 limiter fail-safe (None key) | PASS |
| C-10 _payload_text ReDoS cap (body-then-hint, <0.5s, sentinel survives) | PASS |
| C-11 durable-storage seam | PASS (structural) |
| C-12/C-12b relay_client 4xx→RelayUnavailableError + poll guard | PASS |
| C-15 relay-restart self-heal | PASS (relay-side pin; brain-side = supervisor.py:664 per-tick guarded register) |

All conformance PASS.

## Regression

| Bar | Result |
|---|---|
| R1 kindled_link suite (gating) | **331 passed** (was 264; +67 across m9/m10 + this change) |
| R2 ruff `brain/ relay/ tests/` (gating) | clean |
| R3 full backend (advisory) | **3825 passed**, 1 skipped, 1 xfailed — no collateral break |
| Layer-2 cost/cache | N/A (off chat path) |

## Verdict: CLEAN → done (with one ACCEPTED residual)

Crypto/opacity unchanged (relay never inspects plaintext, never mutates an envelope).
The blocker (store-and-forward) + the cross-tenant rate DoS are closed. **Accepted
residual (deferred to the Phase-7b public-relay checklist):** open `/envelope` + 256
queue-depth lets a host-rate-limited attacker slowly fill a specific victim's inbound
queue (blocks new senders; victim still reads own mail) — bounded, self-healing,
trusted-LAN scoped. Durable storage backend + live cross-machine validation (#51) remain
infra deferrals.
