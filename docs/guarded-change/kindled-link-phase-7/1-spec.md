# Kindled Link Phase 7a — Activation (local-loopback-first, real self-hosted relay)

Stage 1 spec for the `guarded-change` loop. Change slug: `kindled-link-phase-7`.
Parent design: `docs/superpowers/specs/2026-06-15-kindled-to-kindled-design.md` (§9 sessions,
§10 peer prompt, §14 memory-and-wiring, §19 phases). Phases 1–6 are merged to main, DORMANT.

## 0. Scope decision (Hana, 2026-06-21)

Three forks were settled before this spec:

1. **Cut width = local-loopback activation FIRST (7a).** Wire the full loop end-to-end and
   prove it; **defer the cross-platform pairing matrix, abuse/quota hardening, and public-relay
   readiness checklist to a later 7b.**
2. **Safety default = OFF by default, explicit opt-in.** Autonomous peer correspondence ships
   disabled. A user toggle (mirroring persona notes, v0.0.38) is the master switch.
3. **Relay transport = a real self-hosted relay over the network** — not loopback-only.

**Reconciliation of (1) vs (3).** (1)'s option text said "no real network"; (3) overrides the
transport question specifically. Resolved intent: **build the real transport against a
deployable self-hosted relay reachable over the network, but keep the validation/hardening
programme (cross-platform matrix, abuse quotas, public-relay checklist) deferred to 7b.** The
relay in 7a is the existing `relay/dev_relay.py` made self-hostable (real bind + config),
**not** a hardened production relay.

## 1. Problem

Everything is built but **nothing fires on a live relay/supervisor path** — the Organ
Definition-of-Done is unmet (parent §14: "Producer fires on a live relay/supervisor path").
The `SessionEngine`, `PrivacyGate`, `relationship.py`, `RelayClient`, and the protocol codec all
exist and pass tests in isolation, but:

- No supervisor cadence ever calls the session engine, so no draft is ever generated, gated,
  or sent autonomously.
- `RelayClient` is never wired to `protocol.build_envelope`/`verify_and_open` — there is no
  outbound `send_fn` that encrypts+signs+pushes, and no inbound path that fetches+verifies+
  decrypts+stores a real envelope.
- There is no inbound flood cap (parent §9: "bound the number of pending envelopes decrypted/
  processed per poll per peer") — the producer of inbound work does not exist yet, so its bound
  cannot exist yet either.
- The §14 through-path test (inbound peer message → stored transcript → gated response attempt →
  relationship state read) cannot be written because the through-path is not connected.
- A pile of small completion-deferreds from Phases 3–6 are explicitly tagged "Phase 7" and
  belong with the wiring that motivates them (cap atomicity under concurrency, recover's real
  `transcript_summary`, `_check_day` clock, gradual-regression signal wiring, rejection logging,
  `_STAGE_GUIDANCE` familiar/friend keys, lost-path provenance, relay-health surface, redacted
  hold-reason).

Until this lands, **no real peer message can leave a local simulator** and the feature is a
dead organ behind a green test suite — the exact failure mode the project's Organ DoD exists to
prevent.

## 2. Why now / why this shape

- This is the **last phase** of the Kindled-link build (Tier 2 #2). The safety spine (Phase 4
  privacy gate) and relationship maturation (Phase 5) already landed and are stage-blind on the
  user-detail bar; activation does not weaken any of that — it only connects producers to the
  live path.
- **Off-by-default** matches every other autonomous organ's posture (notes shipped OFF). This
  is autonomous network correspondence — the single most consequential autonomy surface in the
  product — so it must not start without a conscious user opt-in, even though the per-peer
  `consent_state == "paired"` gate already exists. The toggle is a second, global gate.
- **Local-loopback-first** de-risks: we prove the entire producer→gate→transport→inbound→
  transcript→relationship loop against a self-hosted relay we control before taking on the
  cross-platform validation and abuse-hardening programme (7b).

## 3. In scope (7a)

0. **Session establishment + key persistence (NEW — stage-3 blocker F1/L2/M1).** The X25519
   handshake (`protocol.derive_session_key`/`build_session_open`/`parse_session_open`) is defined
   but **called only in tests** — no `brain/` code establishes a session, and the store has **no
   session-key column**, so `verify_and_open` (which needs `session_key`+`sender_role`) cannot
   run on a live inbound. This cut wires it:
   - **It is a mutual TWO-message X25519 handshake** (re-review NF1, verified at
     `protocol.py:66-78`): `derive_session_key(my_eph_priv, peer_eph_pub, …)` needs the *peer's*
     ephemeral public, and each `session_open` carries only *one* side's ephemeral. Neither side
     has a key until both ephemerals are exchanged. A 1-RTT key would require a static (non-
     ephemeral) X25519 key, which **defeats per-session forward secrecy** — rejected.
   - Two store tables: `pending_handshakes(peer_id, session_id, my_eph_priv BLOB,
     bootstrap_nonce BLOB, my_role INT, created_at)` holds the initiator's pending ephemeral until
     the reply arrives; `session_keys(peer_id, session_id, session_key BLOB, my_role INT,
     peer_role INT, established_at)` holds the completed key (both mirror `seq_high_water`). Keys
     persist so inbound decrypt survives a supervisor restart (B2/B0-2).
   - **Flow:** initiator mints `session_id`, generates an ephemeral, writes a `pending_handshakes`
     row, pushes a `session_open` (leg 1). Responder receives it, **binds to
     `envelope["session_id"]`** (NU1), generates *its own* ephemeral, derives with the
     *initiator's* `bootstrap_nonce` as HKDF salt (NF3), writes `session_keys` + **calls
     `store.create_session`** (NU2 — else every gated response holds forever on a missing
     `sessions` row → dead organ), and pushes a reply `session_open` (leg 2). Initiator receives
     the reply, derives, writes `session_keys` + `create_session`, clears the pending row. A
     restart mid-handshake reloads the pending row (recovery, §3.5).
   - A message envelope for a (peer, session) with **no stored `session_keys` row** is **dropped**
     (logged `inbound_no_session`), never guessed. A `session_open` control envelope (signed-not-
     encrypted; `"session_open" in envelope`) routes to the handshake, distinguishable *before*
     any key exists (NL2).
   - **Clobber guard (lap-3 F-MAJOR).** `session_open` is signed-not-encrypted with `sequence:0`,
     so the message-path replay high-water does NOT cover it. `on_session_open` rejects (idempotent
     ignore) a `session_id` that already has a `session_keys` row, and `open_session` refuses to
     reuse a live `session_id` — else a replayed leg-1 overwrites a live key.

   **Peer mailbox addressing (lap-3 F-BLOCKER — decoupled-mailbox scheme, Hana 2026-06-21).**
   Nothing maps peer→relay-mailbox today (`create_invite`/`peers`/envelopes carry no mailbox), so
   neither handshake leg is addressable. Fix (privacy-correct, parent design §6.2 intent):
   - **Own mailbox** = a random `mbx_…` id generated once and persisted locally
     (`store.get_or_create_local_mailbox()`), **decoupled from `key_id`** so the relay cannot tie
     a mailbox to an identity. Rotation is 7b.
   - **Invite** (`create_invite`) gains `mailbox_id` (the inviter's own); `import_invite` persists
     it to a **new `peers.relay_mailbox` column**. This addresses **leg-1** (the initiator imported
     the responder's invite → knows the responder's mailbox).
   - **`build_session_open`** gains a signed `sender_mailbox` field; on leg-1 the responder reads
     it and `upsert_peer(relay_mailbox=…)`, learning the initiator's mailbox to address **leg-2**
     and all subsequent messages. (Additive outer field → the protocol KAT vectors regenerate;
     DORMANT/unreleased, no wire-compat lock yet.)
   - Steady-state messages address from the persisted `peers.relay_mailbox` on each side; the
     `RelayClient.push` `relay_mailbox`/`build_envelope` target is sourced from it, never a literal.
   - Forward-secrecy/at-rest posture unchanged from parent §20: the session key is plaintext at
     rest like the transcript + privkey (local-OS-compromise out of scope); the wire stays
     forward-secret via per-session ephemerals.

1. **Supervisor wiring** — exactly the notes pattern (L1 correction): a `kindled_link_enabled`
   **`run_folded` param defaulting True** (a coarse build/test switch), while the **real
   off-by-default user gate is `config.kindled_link_enabled` (default False) read inside the
   tick** — identical to how notes' param defaults True but `brain/notes/__init__.py:31` gates on
   `config.notes_enabled`. A fault-isolated `_maybe_run_kindled_link_tick` block in `run_folded`,
   mirroring the maker/notes tick blocks (`brain/bridge/supervisor.py:479–505`).
   The tick gates itself internally on a **persisted wall-clock cadence**
   (`kindled_link_cadence_state.json`, mirroring `brain/soul/cadence.py` — survives restart/
   sleep, unlike the monotonic timers). The `kindled_link_enabled` flag only switches the block.
2. **Live transport** — an outbound `send_fn` that builds a real encrypted+signed envelope
   (`protocol.build_envelope`) and pushes it via `RelayClient.push`; an inbound poll that calls
   `RelayClient.fetch` → `protocol.verify_and_open` (full §8 reject order, replay high-water via
   `store.get/set_seq_high_water`) → decrypt → `store.append_transcript`, then schedules a gated
   response. Session-open handshake (`build_session_open`/`parse_session_open`,
   `derive_session_key`) on first contact.
3. **Inbound flood cap** — `RelayClient.fetch` returns the **whole mailbox** (`dev_relay.py:33`,
   not per-peer), so the adapter **groups fetched envelopes by `sender_key_id`** (F3) and bounds
   decrypted/processed envelopes per poll per peer (`limits.INBOUND_FLOOD_CAP`); excess left on
   the relay (un-acked) and surfaced as a degraded state. Local defence independent of any
   relay-side quota (7b). **Known residual (U2):** un-acked excess is re-fetched and re-verified
   next poll (REPLAY reject still prevents a second transcript row); the cap bounds per-poll work,
   not cumulative re-verify under a sustained flood — acceptable for the alpha, logged.
4. **Off-by-default opt-in toggle** — `PersonaConfig.kindled_link_enabled: bool = False`
   (mirror `notes_enabled` at `brain/persona_config.py:84`, incl. defaults dict + `from_dict`);
   `POST /persona/config/kindled-link` (auth-gated, strict-validated); a Connection-panel toggle
   mirroring `NotesToggle.tsx`. Autonomous starts are suppressed unless the flag is True AND the
   peer is `paired` AND not paused/revoked (the existing `can_start_session` gate).
5. **Recovery banner** — on supervisor start the engine reloads peer/session/cursor state and
   `recover()` re-gates half-finished drafts (already implemented); when state was recovered the
   `Kindled Links` panel shows a recovery banner (parent §9, consistent with other recovered-
   state surfaces). Surfaced via a view field on `GET /kindled-link/peers` or a small status
   endpoint.
6. **Self-hostable relay** — `relay/dev_relay.py` made runnable as a self-hosted process (real
   host/port bind via env/CLI, `__main__` entry, README run-note). No durable storage upgrade,
   no quotas, no abuse hardening (7b). In-memory mailbox is acceptable for the alpha. **Bind
   constraint (U4):** `/envelope` push is unauthenticated (`dev_relay.py:96`), so 7a binds
   **loopback or a trusted LAN only** — the run-note states this and §7's "no public exposure"
   depends on it. A public bind waits for 7b abuse hardening.
7. **§14 through-path test** — a real inbound envelope (built with the test peer's identity,
   pushed through the live relay client against a `dev_relay` instance) leads to: stored
   transcript row, a gated response **attempt** (gate is consulted; with the default
   conservative posture the attempt may `hold` — the test asserts the gate ran and a transcript/
   relationship read occurred, not that a send happened), and a relationship-state read.
8. **Drained Phase-7 deferreds (the safe/small ones):**
   - **Cap atomicity (Phase 4 carry).** The tick is the first place per-peer provider/outbound
     cap reads and increments run under potential concurrency. Serialise per-peer cap
     check-then-increment (a per-peer lock or a single-writer tick) so 60/day cannot be exceeded.
   - **`recover` real `transcript_summary` (Phase 3 carry).** `recover()` currently passes
     `transcript_summary=""`; wire the real recent-transcript summary.
   - **`_check_day` clock (Phase 4/5 carry).** Resolve the UTC-vs-local `today` coupling at the
     live tick boundary — one authoritative clock source for `now`/`today`.
   - **Gradual-regression signal wiring (Phase 5 carry).** Feed hold-log frequency and peer
     emotion-pressure into `run_relationship_reflection` as the gradual −1 regression signal.
   - **Ungrounded-quote rejection logging (Phase 5 carry).** Log rejected ungrounded reflection
     quotes (audit-tier JSONL), mirroring attunement's rejection log.
   - **Gate `familiar` self-disclosure tier (Phase 5 carry; F2 naming correction).** The carry
     was mis-labelled "`_STAGE_GUIDANCE` keys": the gate's self-disclosure latitude is the
     hardcoded tuple `stage in ("friend","close")` at `privacy_gate.py:78` (no map), and the
     `_STAGE_GUIDANCE` map in `peer_prompt.py:24` is a different (peer-prompt) artifact also
     missing `familiar`/`friend`. The real work: add a **mild `familiar` self-disclosure tier**
     to `_build_gate_prompt` (so `familiar` no longer falls through to the strictest path) while
     the **user-detail bar stays provably stage-blind** (re-assert the Phase-5 byte-diff/semantic
     equivalence property). The `peer_prompt` `_STAGE_GUIDANCE` gap is documented but the
     load-bearing fix is the gate tier.
   - **Lost-path provenance (Phase 5 carry).** Extend `_peer_attributed` to the lost/graveyard
     recall path (the tripwire-pinned gap).
   - **Relay-health surface (Phase 6 carry).** Surface relay reachability (last successful
     poll/push, `RelayUnavailableError` state) on the panel.
   - **Redacted hold-reason (Phase 6 carry).** The holds line shows a redacted reason category
     (never the held body — the Phase-6 safety spine holds).

## 4. Out of scope (deferred to 7b — logged in the deferred ledger)

- Cross-platform pairing validation (mac/win/linux real-machine matrix).
- Abuse & quota hardening (relay-side rate limits, durable relay storage, nonce TTL GC).
- Public-relay readiness checklist + official relay operations.
- **Live-spectator WS streaming + delivery-state** (Phase 6 carry). The panel already shows the
  transcript via the GET poll; real-time WS streaming is a UI nicety, not activation-core.
- **Redacted hold-reason category** (deferred during build, 2026-06-22). The holds COUNT +
  session_id/created_at already surface (Phase 6 spine intact); a coarse redacted reason category
  per held draft is marginal observability → 7b.
- OS keychain / at-rest encryption (parent §20, unchanged).

## 5. Invariants this change must NOT break (carried from Phases 3–6)

- **No peer-derived content can flip hold/revise → send** (parent §5; Phase-4 spine). The
  transport and tick add inputs (inbound transcript, relationship stage) that already flow only
  into the *fenced LLM prompt*, never into the pre-filter or budget arithmetic. Activation must
  preserve this: nothing peer-sourced may reach a send decision except through the gate.
- **Tool-less peer path.** Drafts are `provider.complete` only; no tool schema / `tool_loop` /
  `reach_for_capability` reachable from peer text (Phase-3 AST conformance oracle).
- **Held-draft body never reaches any `/kindled-link/` response** (Phase-6 spine: `holds_status`
  projects only `session_id`+`created_at`). New endpoints/fields must not leak the body.
- **Provenance:** `kindled_peer` memories stay peer-marked at every recall render point.
- **Background discipline:** every peer provider call takes a `cli_throttle` background slot and
  yields to interactive chat; peer work counts against the shared global daily budget.
- **Caps:** 60s outbound gap, 24/session, 6h session cooldown, 20/day outbound, 60/day provider,
  inbound flood cap — all enforced on the live path, not just unit-tested.
- **State recovery:** half-finished outbound is re-gated, never blind-resent; a draft not
  currently sendable is deferred, not dropped.

## 6. Wiring (parent §14 — source of truth for promotion review)

**Reads-from:** peer transcript store, relationship state, local ambient (emotion/body/felt-
time/interior via the existing peer-prompt composer), relay (inbound envelopes),
`PersonaConfig.kindled_link_enabled`, persisted cadence state.

**Feeds-into:** (1) peer ambient — future sessions get continuity; (2) memory — `kindled_peer`
typed memories age/fade/recall; (3) emotion — capped, vocab-filtered, decay-subordinate peer
deltas (existing `apply_peer_emotion`); (4) feed — pair/session/milestone events (existing
`feed_source`); (5) relationship state — reflection updates stage/trust/boundaries, now fed the
gradual-regression signal.

**Does NOT feed:** user attunement, user-presence computation, or initiate-to-user (unchanged).

**Producer fires on:** the supervisor `_maybe_run_kindled_link_tick` live path. **Readers:**
peer prompt reads relationship state; feed reads milestone events; memory/forgetting reads typed
peer memories; the panel reads transcript/holds/relay-health.

## 7. Risks

- **Activation is hard to undo socially** — once she sends a real message to a peer, it is sent.
  Mitigated by off-by-default + paired-only + caps + the gate. The relay is self-hosted/alpha;
  no public exposure.
- **Concurrency on caps** — the tick must not race itself or a future second consumer past the
  60/day cap. Addressed by the cap-atomicity item.
- **Relay outage** — bounded retries + backoff already in `RelayClient`; the tick must treat
  `RelayUnavailableError` as a degraded state (surface it), never a crash (fault-isolated block).
- **Regression risk to the shared subscription** — a new always-on background consumer. Gated by
  off-by-default + `should_yield` + the shared daily budget; advisory cost metrics watched at
  stage 8.

## 8. Acceptance criteria

See `1.5-criteria.md` (the conformance oracle for stage 8).
