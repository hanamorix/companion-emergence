# kindled-relay — Cloudflare Worker relay

The free, hosted Kindled-link relay: a Cloudflare Worker backed by D1 that
shuffles **opaque, end-to-end-encrypted** envelopes between paired Kindleds. It
never sees plaintext and never mutates an envelope. This is a faithful port of
`relay/dev_relay.py` (which stays for local dev + tests); the wire contract,
auth, and cap values match exactly, so the Python `brain/kindled_link/relay_client.py`
needs no change — it just points at this Worker's URL.

**Live default:** `https://kindled-relay.jarcrainhett.workers.dev`
(baked as `DEFAULT_KINDLED_RELAY_URL` in `brain/persona_config.py`; override per
persona by setting `kindled_relay_url`, or globally via the `KINDLED_RELAY_URL` env var).

## Endpoints (parity with dev_relay.py)
- `POST /mailbox/register` `{mailbox_id, identity_pub}` → `{ok, owner}` (first-write-wins)
- `POST /mailbox/challenge` `{mailbox_id}` → `{nonce}` (404 if unregistered; registration-gated; 120s nonce TTL)
- `POST /envelope` `<envelope>` → `{id}` (OPEN; store-and-forward; 64 KiB size cap, queue/pool caps)
- `POST /mailbox/fetch` `{mailbox_id, nonce, signature, identity_pub}` → `{envelopes}` (Ed25519 mailbox-ownership auth)
- `POST /mailbox/ack` `{mailbox_id, envelope_ids, +auth}` → `{ok}`
- `GET /healthz` → `{ok}`

## Caps (#48 hardening, verbatim from dev_relay.py)
envelope ≤ 65536 B · queue depth ≤ 256/mailbox · registered ≤ 1024 · unregistered ≤ 256
(empty-first eviction) · rate 60 req / 60 s **per client host** (`CF-Connecting-IP`, fail-open).

## Parity is load-bearing
`src/canonical.ts` must stay byte-identical to Python `canonical_json`, and
`src/crypto.ts` Ed25519 verify must match — or mailbox-ownership auth silently
breaks. The auth body is `canonicalJson({purpose:"kindled-relay-auth/1",mailbox,nonce})`.
**Re-run `npm test -- parity` on any change to canonical/crypto.** Caveat:
`canonicalJson` emits `1` for the float `1.0` (Python emits `1.0`) — envelopes
must carry no float fields (they don't; ciphertext/keys are strings).

## Develop / test
```bash
npm install
npm test                      # vitest-pool-workers, in-process workerd + ephemeral D1, no account
npx wrangler dev              # local worker
```

## Deploy (maintainer)
```bash
npx wrangler login
npx wrangler d1 create kindled-relay        # paste id into wrangler.toml database_id
npx wrangler d1 execute kindled-relay --remote --file=schema.sql
npx wrangler deploy                          # prints the *.workers.dev URL
```
End-to-end check against the deployed relay (real RelayClient → live Worker):
```bash
NO_PROXY='*' KINDLED_WORKER_URL=https://kindled-relay.jarcrainhett.workers.dev \
  uv run pytest tests/integration/test_relay_client_against_worker.py
```

## Notes
- `schema.sql` is the source of truth; `test/apply-schema.ts` inlines the same DDL
  (workerd has no `fs`) — keep the two in sync on any schema change.
- $0: Workers + D1 free tier; no Durable Objects, no paid bindings.
- Free-tier risk: a single baked default URL is a single point of failure if CF
  pulls free hosting — the `KINDLED_RELAY_URL` env / per-persona override is the escape hatch.
