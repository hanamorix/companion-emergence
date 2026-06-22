# Stage 3 — Cold plan re-review (Phase 7a, after bounce #1)

Fresh cold subagent over the revised {1-spec, 1.5-criteria, 2-plan} + prior review. Main-thread
spot-verified the handshake-math crux (`protocol.py:66-81, 207-252`).

## Verdict
**Worst severity: BLOCKER (re-opened, sharper root).** Route → stage 1. The `session_keys` table
closes "nowhere to store it," but the revision's handshake *topology* is crypto-impossible.

## Blocker-closed? — NO (new root)
`derive_session_key(my_eph_priv, peer_eph_pub, …)` (`protocol.py:66-78`) **requires the peer's
ephemeral public key**. `build_session_open` carries only the *sender's own* ephemeral
(`:213, :230`); `parse_session_open` returns only that one (`:249-252`). So the initiator **cannot
derive — or persist — a session key when it sends `session_open`** (B0-1/T2.5 assert it does). It
is a **mutual two-message exchange**: initiator sends its ephemeral → responder generates its own,
derives+persists, **replies with its ephemeral** → initiator derives+persists. The integration
test `tests/kindled_link/test_phase2_integration.py:52-61` only works because it holds both
ephemerals synchronously. The `session_keys` schema has no slot for the initiator's *pending
ephemeral private key*, so it can't even represent the intermediate state.

## New findings
- **NF1 — BLOCKER — initiator can't derive at open time; schema can't hold the pending ephemeral.**
  Fix: a `pending_handshakes(peer_id, session_id, my_eph_priv BLOB, bootstrap_nonce BLOB, my_role,
  created_at)` row; responder reply leg; initiator completion leg.
- **NF2 — MAJOR — `accept_session_open` is half a handshake** — no responder reply step, so the
  initiator never gets `eph_b` and stays keyless.
- **NF3 — MINOR — bootstrap_nonce symmetry unstated** — responder must reuse the *initiator's*
  nonce as HKDF salt (`:81`), not mint its own.
- **NL2 — MAJOR — T4 session-open routing is right but reply-leg absent.** The adapter *can*
  branch on `"session_open" in envelope` before having a key (signed-not-encrypted) — good — but
  must emit a responder reply / complete the initiator side.
- **NU1 — MAJOR — `session_id` allocation unstated.** `build_session_open`/`create_session` both
  take it as a caller arg; nothing generates it. Responder MUST bind to `envelope["session_id"]`
  or `(peer_id, session_id)` PKs diverge and replay protection breaks.
- **NU2 — MAJOR — handshake must call `store.create_session`.** `process_outbound`/`_send_allowed`
  hold if no open `sessions` row (`session_engine.py:129-131`). If the handshake writes a
  `session_keys` row but no `sessions` row, every gated response holds forever — a green-test dead
  organ (A2 "a hold satisfies A2" would mask it). Wire `create_session` into completion.
- **NU4 — MINOR — restart mid-handshake** must reload the pending row (§3.5 recovery only covers
  drafts/cursor).

## Recommended fix shape (NM1b — crypto-exact, ~25 lines)
`pending_handshakes` table → `open_session` persists pending ephemeral + returns the session_open
to push → on inbound `session_open`: responder generates ephemeral, `derive_session_key`, writes
`session_keys` + `create_session`, replies session_open; initiator on the reply derives, writes
`session_keys` + `create_session`, clears pending. session_id minted by initiator, adopted by
responder from the envelope. Reject 1-RTT-via-static-key (defeats per-session forward secrecy).

## Prior findings confirmed addressed
F2 ✓ (gate `:78` tuple), F3 ✓ (group-by sender_key_id), F4 ✓ (new reflection param — confirmed
none today), L1 ✓ (param True/config False), L3 ✓ (`UPDATE…RETURNING`; sqlite 3.42/py3.12 support
confirmed), M1 ◐ partial (table added but can't hold pending ephemeral — NF1), M2 ✓ (cadence
helpers `relationship.py:214-242` exist), M3 ✓ (T12 verify-first), U2 ✓ (logged residual),
U4 ✓ (loopback bind).

## Spot-verification (main thread)
`derive_session_key(my_eph_priv, peer_eph_pub, …)` needs peer ephemeral — **CONFIRMED**
(`protocol.py:66-78`). `build_session_open` carries one ephemeral; `parse_session_open` returns
one — **CONFIRMED** (`:213,230,249-252`). The two-message-handshake conclusion holds.
