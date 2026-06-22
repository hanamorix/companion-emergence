# Stage 3 — Cold plan re-review, lap 3 (Phase 7a session layer)

Tightly scoped to the corrected 3-leg handshake. Main-thread spot-verified pairing/peers schema.

## Verdict
**Crypto/role layer: CORRECT (prior two blockers genuinely closed).** **New BLOCKER one layer
down: peer mailbox addressing is undefined** — neither handshake leg can be addressed on a live
relay. Route → stage 1 (a design decision, surfaced to Hana).

## Confirmed CORRECT now
- **Roles/nonce (Q1):** `session_keys` stores `my_role`+`peer_role`; send uses `my_role`, receive
  uses `peer_role` as `sender_role`. Nonce = `role_byte‖seq` → initiator frames `0x00…`, responder
  `0x01…` → disjoint nonce space under the shared key even at equal sequence. No reuse.
- **2-message topology:** `pending_handshakes` holds the initiator ephemeral until the reply;
  both derive the same key (sorted-fp HKDF info, shared bootstrap_nonce salt). Sound.
- **Leg-1/leg-2 disambiguation (Q4):** responder has no pending row → `on_session_open`;
  initiator has one → `complete_session`. Coherent, no dual-state case.

## New findings
- **F-BLOCKER — reply-leg (and leg-1) mailbox addressing undefined.** `push` files into
  `mailboxes[envelope["relay_mailbox"]]` = the **recipient's** mailbox (`dev_relay.py:23-30`,
  `relay_client.py:44-52`). The responder learns the initiator only from leg-1, whose
  `relay_mailbox` is the responder's own; `parse_session_open` returns only `{ephemeral_pub,
  bootstrap_nonce}` (`protocol.py:249-252`); the envelope has `sender_key_id` but **no
  sender_mailbox** (`build_session_open:220-233`). The peer record can't supply it:
  `create_invite` body has `relay_url`/`identity_pub`/`fingerprint` but **no mailbox**
  (`pairing.py:30-37`); `peers` table has `relay_url` but **no mailbox column** (`store.py:24-32`).
  → leg-2 is unaddressable; leg-1 too (initiator has no responder mailbox). **Spot-verified.**
  Note: `peer_id == fingerprint == key_id` (`pairing.py:62,35`), so the **key_id is always known**
  — only the *mailbox* mapping is missing.
- **F-MAJOR — Q3: no session_id-clobber guard.** `session_open` is signed-not-encrypted,
  `sequence:0`, so the message-path REPLAY high-water doesn't cover it; `parse_session_open` has no
  idempotency check. A reused leg-1 `session_open` with an in-use `session_id` could overwrite a
  stored `session_key`. Fix: `on_session_open` rejects/idempotently-ignores a `session_id` that
  already has a `session_keys` row; `open_session` refuses to reuse a live session_id.
- **F-MAJOR — B0-1/A1 can't detect the routing blocker** (both drive the handshake in one process
  with mailbox literals in scope). The oracle must require a **bidirectional** completion where
  each side derives its push target from stored/received state, not a hardcoded literal.

## Citations
`protocol.py:66-97,124-192,207-252`; `store.py:24-42,208-236`; `pairing.py:26-69`;
`relay_client.py:21,40-52`; `dev_relay.py:23-37,61-78`; `identity.py` (key_id only, no mailbox);
parent `design.md:149` (invite "encodes relay mailbox capability" — specified, NOT implemented).

## Unverifiable
Whether `mailbox_id == key_id` is an intended unwritten convention — not stated anywhere; would
close the gap for free but trades metadata privacy (stable per-identity mailbox = relay can
correlate all of a Kindled's correspondence). A design call for Hana.
