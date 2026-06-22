# Stage 3 — Cold plan red-team (Phase 7a)

Cold independent subagent (no shared context) over {1-spec, 1.5-criteria, 2-plan} + the Layer-2
`redteam_context` source. Four lenses, ranked, cited. Main-thread spot-verified the blocker +
flood-cap + relay citations (all confirmed — see end).

## Verdict
**Worst severity: BLOCKER.** Recommended route: **→ stage 1 (spec)** to add session
establishment + session-key persistence as explicit in-scope work before the through-path (A1)
or replay-across-restart (B2) can connect.

## Factual
- **F1 — BLOCKER — No `session_key` is persisted anywhere; inbound decryption cannot connect.**
  `verify_and_open` requires `session_key: bytes` + `sender_role: int` (`protocol.py:156-165`).
  The only producers — `derive_session_key`/`build_session_open`/`parse_session_open`
  (`protocol.py:66-81, 207-252`) — are called **only in tests** (`test_phase2_integration.py`,
  `test_protocol_reject.py`, `test_protocol_kat.py`), **never in any `brain/` module**. The store
  schema (`store.py:24-93`, 10 tables) has **no** session-key/role/ephemeral column. After a
  restart (B2 requires reload-from-store) there is no key to reload → inbound decrypt is
  structurally impossible. Plan §3.2 mentions the handshake in one clause and T4 sources only
  `seq_high_water` — silent on where `session_key`/`sender_role` come from. The parent design §9
  line 324 also under-specified this ("reloads peer/session/cursor", no key); the plan inherits
  the hole and asserts the through-path "connects as designed."
- **F2 — MAJOR — H1/`_STAGE_GUIDANCE` mis-named + wrong module.** Criteria H1 + spec §3.8 name
  `_STAGE_GUIDANCE` familiar/friend keys. The real `_STAGE_GUIDANCE` is in `peer_prompt.py:24-36`
  (keys stranger/acquaintance/close) — the **peer-prompt** composer, not the gate. The gate's
  latitude is the hardcoded tuple `stage in ("friend","close")` at `privacy_gate.py:78`, no map,
  no `familiar` key. T10 targets the right function but the criterion mis-names the artefact.
  `familiar` is a real reachable stage (`relationship.py:23`) currently falling through to the
  strict path — motivation sound.
- **F3 — MINOR — flood cap "per peer" vs whole-mailbox fetch.** `relay_client.fetch` →
  `dev_relay.fetch` returns the **whole mailbox** (`dev_relay.py:33`), all senders, no partition.
  Per-peer clamp (C1/C2) must group by `sender_key_id` (envelope field) before clamping; the
  plan implies but never names the grouping key. One shared mailbox per persona → a flooding peer
  could starve a quiet peer without explicit grouping.
- **F4 — MINOR — H4 regression-signal has no seam.** `run_relationship_reflection`
  (`relationship.py:139-208`) computes regression purely from model JSON; **no parameter** for an
  external hold-frequency/emotion-pressure signal. T11 must add the param + tick-side computation;
  the plan reads as if the seam exists.
- **Verified clean:** `get/set_seq_high_water` (`store.py:208-226`), `recover()` passes
  `transcript_summary=""` today (`session_engine.py:262-266`), `_check_day` (`:25-36`),
  `append_transcript`/`get_pending_drafts`/`save_draft` (`store.py:325-363`), off-by-default
  mirror accurate (`persona_config.py:52,84,130` + `notes/__init__.py:31`).

## Logical
- **L1 — MAJOR — spec §3.1 contradicts T7 on the off-by-default gate.** §3.1 says the new flag is
  "DEFAULT False, unlike notes which default True." But notes' `run_folded` **param** defaults
  True (coarse switch); the real gate is `config.notes_enabled` (False) inside the tick. One
  consistent answer: **param defaults True (mirror), config field defaults False (real gate)**.
- **L2 — MAJOR — §14 through-path A1/A3 unsatisfiable until F1 resolved.** A3 demands the test
  drive `_run_kindled_link_tick` → inbound adapter → `verify_and_open`. Without a persisted/
  derived key the test can only pass by injecting the key itself — the "hand-assembled engine"
  A3 forbids. A2 (spy gate+relationship on outbound) passes independently — green A2 over a dead
  A1 is exactly the DoD failure.
- **L3 — MINOR — cap atomicity: single-writer-tick insufficient vs two-supervisor.** `incr_*`
  commits per-statement with no check-then-increment transaction (`store.py:306-313`). The
  v0.0.37 double-bridge case races. T8 should use a store-level atomic
  increment-and-return (`UPDATE … RETURNING`), not a lock-by-convention.

## Missed opportunity
- **M1 — MAJOR — add a `session_keys(peer_id, session_id, session_key, my_role, established_at)`
  table** written on `parse_session_open` (responder) / `build_session_open` (initiator) — ~15
  lines mirroring `seq_high_water`. Closes F1 without redesign. The plan's file-structure omits
  any session-key persistence change — the single most load-bearing missing piece.
- **M2 — MINOR — reuse the existing `relationship.py:211-242` persisted-cadence helpers** instead
  of a new `cadence.py` (two cadence idioms in one package). Or state why soul-style relay-error
  backoff needs the richer helper.
- **M3 — MINOR — H2 lost-path shape mismatch.** The lost render path (`prompt.py` ~657) renders
  from a graveyard **dict** (`entry.get("graveyard_reason")`), not a `Memory`; `_peer_attributed`
  reads `mem.memory_type`. The graveyard entry must carry `memory_type` or H2 passes a stubbed
  test while the real graveyard drops the type.

## Unstated assumptions & risks
- **U1 — (core of F1) — assumes a session_key materialises at the inbound boundary.** Open
  question for stage 1: **who initiates `session_open`, when, and where is the key written?**
- **U2 — MINOR — un-ack-and-leave re-verifies every poll.** `dev_relay.fetch` returns whole
  mailbox un-acked; flood-excess is re-fetched + re-`verify_and_open`'d each poll (REPLAY reject
  protects transcript, not crypto work). Flood cap bounds per-poll, not cumulative re-verify.
- **U3 — MINOR — assumes `provider.complete` logs to `chat_usage.jsonl`.** Unverified; if not,
  the advisory metric is silently empty (advisory-only, non-gating).
- **U4 — MINOR — open `/envelope` on a network bind.** `dev_relay.py:96-98` push is
  unauthenticated; a real network bind lets anyone push to any registered mailbox. 7a must bind
  **loopback/trusted-LAN only**, or §3.6 contradicts §7's "no public exposure."

## Spot-verification (main thread)
- F1: handshake callers grep → only `tests/`; `brain/`/`relay/` = zero. **CONFIRMED.**
- F1: `store.py` schema = 10 tables (peers/consumed_invites/seq_high_water/sessions/peer_counters/
  outbound_drafts/transcript/disclosure_budget/relationship_state/peer_emotion_window) — no key
  column. **CONFIRMED.**
- F3/U4: `dev_relay.fetch` returns `mailboxes.get(mailbox_id, [])` whole; `/envelope` open.
  **CONFIRMED.**
