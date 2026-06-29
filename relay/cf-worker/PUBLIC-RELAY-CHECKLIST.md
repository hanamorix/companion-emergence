# Public Relay Operational Checklist

Operational notes for running `kindled-relay` as a public Cloudflare Worker relay.

---

## Rate limiting

**The CF native ratelimit binding is per-isolate best-effort, NOT a hard global ceiling.**

Cloudflare's native ratelimit (`[[unsafe.bindings]]` type `ratelimit`) is enforced per Worker isolate. Under high traffic, multiple isolates run concurrently — each with its own counter — so the effective global rate may temporarily exceed the configured `60 req / 60s` per key.

For a hard global cap you would need a coordinated store:
- **Durable Objects** (the canonical CF solution) — paid plan required.
- **D1 counter** (free, but non-atomic reads → races under burst) — not recommended for strict enforcement.

Documented limitation: the current free-tier rate limiting is a best-effort guard against casual abuse, not a strict hard ceiling.

---

## D1 storage budget

**D1 free tier = 5 GB.**

Worst-case full-queues calculation:
- Max envelope size: 64 KiB
- Max queue depth per mailbox: 256
- Max registered mailboxes: 1 024
- Max unregistered mailboxes: 256
- Theoretical max: 64 KiB × 256 × (1 024 + 256) ≈ **21 GB** — exceeds the free tier.

Mitigations in place:
1. **Per-mailbox queue cap** (256 envelopes): a single mailbox cannot fill D1.
2. **7-day envelope GC** (`ENVELOPE_TTL_MS`): undelivered envelopes older than 7 days are deleted every 6 hours.
3. **Empty-mailbox eviction** (`evictOldestEmptyUnregistered`): pressure-triggered on push to a new mailbox when the unregistered pool is full.

**Hard aggregate-size guard is DEFERRED** — revisit if D1 usage climbs toward the free-tier ceiling (see ledger #48 storage half). Monitor via the Cloudflare dashboard → D1 → `kindled-relay` database size.

---

## GC cadence

A Cloudflare Cron Trigger fires `scheduled()` **every 6 hours** (`0 */6 * * *`).

Each run sweeps:
- `envelopes` older than 7 days (`queued_at < now - ENVELOPE_TTL_MS`)
- `nonces` with `expires_at < now` (belt-and-suspenders; nonces are also swept lazily on `issueNonce`)
- `mailboxes` that are unregistered, empty (no envelopes), and older than 7 days

The GC summary `{"envelopes": N, "nonces": N, "mailboxes": N}` is logged to `console.log` and appears in `wrangler tail`.

To watch live:
```
wrangler tail --env production
```

---

## Deployment notes

- `wrangler 3.x` emits deprecation warnings about `unsafe.bindings` for the ratelimit binding — these are cosmetic and do not affect runtime behaviour.
- The controller deploys via `wrangler deploy` — do NOT run it manually from a worktree.
- Schema migrations: apply via `wrangler d1 execute kindled-relay --file schema.sql --remote` before deploying code that requires new columns or tables.

---

## Monitoring checklist (post-deploy)

- [ ] `wrangler tail` shows the Worker responding to requests (no unhandled errors)
- [ ] A cron run appears in `wrangler tail` within 6 hours of deploy
- [ ] D1 database size (Cloudflare dashboard) is under 4 GB (leave headroom)
- [ ] `/healthz` returns `{"ok":true}` from the deployed URL
