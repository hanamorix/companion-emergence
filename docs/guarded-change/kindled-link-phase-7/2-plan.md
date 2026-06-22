# Phase 7a — Plan (how + measurement + instrumentation + thresholds)

Stage 2 of the `guarded-change` loop. Derives from `1-spec.md` + `1.5-criteria.md`. TDD per the
project hard rule: each task opens with a failing test. Subagent-driven build (Sonnet) with the
Phase-3..6 cadence. Branch: `feat/kindled-link-phase-7`.

## Approach

Connect the already-built, already-tested organs to the live supervisor + relay path, **off by
default**, behind the existing per-peer `paired` gate plus a new global user toggle. Add only the
**seam code** that was deferred precisely because no live path existed: the tick, the transport
adapters (encrypt+push / fetch+verify+decrypt+store), the flood cap, the toggle, the recovery
surface, and the small Phase-5/6 completion items that the wiring motivates.

**No new model entry points.** Drafts/gate/revision/reflection still go through the existing
tool-less `provider.complete`. The tick adds orchestration + I/O, not new agency.

## Measurement & instrumentation (the methodology's "instrument before you build")

- **New instrumentation: `brain/kindled_link/audit.py` → `<persona_dir>/kindled_link/transport.jsonl`**
  (audit-tier, append-only, streaming-reader friendly). One row per transport event:
  `{ts, event, peer_id, session_id?, seq?, reject_reason?, count?, relay_ok}` where `event ∈
  {poll, push, inbound_accepted, inbound_rejected, flood_clamped, relay_unavailable}`. This is
  the signal that makes C (flood cap), B3 (reject reasons), and the relay-health surface
  **measurable**, and is the conformance evidence for stage 8. Joins the project's standing
  JSONL bounded-tail retention defer (no per-log policy in 7a).
- **Conformance measurement (stage 8):** run the §14 through-path + flood + off-by-default tests
  and read `transport.jsonl` + transcript rows to confirm A/B/C/D/G against `1.5-criteria.md`.
- **Regression measurement (stage 8):** **advisory only** per the Layer-2 config — no comparable
  replay workload exists, and peer calls are background `generate` rows with no fixed workload.
  Read `chat_usage.jsonl` tail for cost/turns/cache deltas; **off-by-default means the standing
  persona shows zero new load**, so the gating chat metrics should be flat. Surface, don't bounce.

## Severity → routing thresholds (gates 4/7/8)

- **Blocker** (→ stage 1): the through-path can't connect as designed; the off-by-default gate
  doesn't actually suppress outbound; a peer-content→send flip is reachable; the tool-less
  invariant is breakable; a held body can leak through a new field.
- **Major** (→ stage 2 at gate 4, → stage 5 at gate 7): cap atomicity not actually serialised;
  flood cap bounds the wrong thing (acks excess / processes unbounded); recovery drops a draft;
  transport adapter mis-orders the §8 reject rules.
- **Minor** (fix in place): a missing test seed, a non-load-bearing naming/log gap, a doc drift.
- **Nitpick** (log): style/clarity.

## Gating vs advisory metrics

- **Gating (conformance, always):** every automated criterion A1–H4 + I1 (full suite/ruff/build).
- **Advisory (surfaced, never auto-bounce):** `cost_per_chat_call_usd`, `num_turns_per_chat_call`,
  `cache_*` deltas (no comparable workload — Layer-2 config). I2 rubric is a human judge call.

## File structure

- **New:** `brain/kindled_link/session.py` (session establishment: initiator `open_session`
  [ephemeral → `build_session_open` → `derive_session_key` → persist], responder
  `accept_session_open` [`parse_session_open` → derive → persist], `load_session_key`); `brain/
  kindled_link/tick.py` (the supervisor entry: cadence gate, poll, inbound ingest, response
  scheduling, autonomous-start gate, recovery — the live path); `brain/kindled_link/transport.py`
  (the `send_fn` builder + inbound `fetch→group-by-sender→verify_and_open→decrypt→store` adapter,
  flood-cap clamp); `brain/kindled_link/audit.py` (transport log).
- **Reuse, do not recreate (M2):** the persisted-cadence helpers already in
  `relationship.py:211-242` — **extend** them (or factor into a shared `cadence` helper) for the
  tick's relay-error backoff; do **not** spawn a parallel `brain/soul/cadence.py` clone.
- **Modify:** `brain/kindled_link/store.py` (**NEW `session_keys` table** + `save_session_key`/
  `get_session_key` mirroring `seq_high_water`; **atomic increment-and-return** for the cap
  counters — `UPDATE … RETURNING` so check-then-increment can't race a second supervisor (L3);
  real `transcript_summary` source); `brain/bridge/supervisor.py` (`kindled_link_enabled` param
  defaulting True + `_maybe_run_kindled_link_tick` block); `brain/persona_config.py`
  (`kindled_link_enabled: bool = False`); `brain/bridge/server.py` (`POST /persona/config/
  kindled-link` + relay-health/recovery view fields); `brain/kindled_link/limits.py`
  (`INBOUND_FLOOD_CAP`); `brain/kindled_link/session_engine.py` (real `transcript_summary` in
  `recover`; caps now use the store's atomic counter; single clock); `brain/kindled_link/
  privacy_gate.py` (`familiar` latitude tier in `_build_gate_prompt`, user-bar stays stage-blind);
  `brain/kindled_link/relationship.py` (**new external-signal param** on
  `run_relationship_reflection` for hold-frequency/emotion-pressure (F4); rejection-quote
  logging); `brain/chat/prompt.py` (`_peer_attributed` on the lost/graveyard render point — and
  ensure the graveyard entry carries `memory_type`/peer-marker so attribution is real not stubbed
  (M3)); `relay/dev_relay.py` (loopback/trusted-LAN bind only — U4); `brain/kindled_link/
  views.py` + `feed`/panel (relay-health + redacted hold-reason + recovery banner).
- **Modify relay:** `relay/dev_relay.py` (self-hostable: host/port via env/CLI, `__main__`,
  run-note). **Frontend:** `KindledLinkToggle.tsx` (mirror `NotesToggle.tsx`) + panel relay-health
  + recovery banner.
- Tests under `tests/kindled_link/` + `tests/bridge/` (mirror prior phases).

## Tasks (TDD; each: failing test → impl → suite green → commit)

**T1 — `kindled_link_enabled` config + endpoint (off-by-default).** `PersonaConfig.kindled_
link_enabled=False` (defaults dict + `from_dict`, mirror `notes_enabled`). `POST /persona/config/
kindled-link` auth-gated + strict bool. Criteria: D2, D3. *(No autonomy yet — pure switch.)*

**T2 — `INBOUND_FLOOD_CAP` + `audit.py`.** Add the limit const + the transport audit logger.
Criteria: instrumentation for C, relay-health, B3.

**T2.4 — Peer mailbox addressing (decoupled-mailbox scheme) — lap-3 blocker fix. Build BEFORE
T2.5.** (a) `store.get_or_create_local_mailbox()` — random `mbx_`+token_hex, persisted once
(single-row table / json), decoupled from `key_id`. (b) `peers.relay_mailbox` column +
`upsert_peer(relay_mailbox=…)`; `create_invite` body gains `mailbox_id` (own); `import_invite`
persists it. (c) `protocol.build_session_open` gains a signed `sender_mailbox` field (inside the
signed outer) — **regenerate the affected KAT vectors** in `tests/kindled_link/test_protocol_kat.py`
and the protocol doc §10 (additive; DORMANT/unreleased so no wire-compat lock). Tests: invite
round-trips the mailbox into `peers.relay_mailbox`; a parsed `session_open` yields the sender's
mailbox; the local mailbox is stable across reload + decoupled from key_id. Criteria: B0-4.

**T2.5 — Session layer (`session.py` + two store tables) — the stage-3 blocker fix (TWO-message
handshake).** New `pending_handshakes(peer_id, session_id, my_eph_priv BLOB, bootstrap_nonce BLOB,
my_role INT, created_at)` + `session_keys(peer_id, session_id, session_key BLOB, my_role INT,
peer_role INT, established_at)` tables + save/get/clear helpers (mirror `seq_high_water`).
`session.py` (three legs):
- `open_session(store, identity, peer, now)` — initiator: refuse to reuse a live `session_id`
  (clobber guard); mint `session_id`, generate ephemeral, write `pending_handshakes`, return the
  leg-1 `session_open` (carrying own `sender_mailbox`) to push at `peers.relay_mailbox`. **No key
  yet.**
- `on_session_open(store, identity, peer, envelope, now)` — responder: **reject/idempotent-ignore
  if `session_id` already has a `session_keys` row** (clobber guard); else bind to
  `envelope["session_id"]`, `upsert_peer(relay_mailbox=envelope.sender_mailbox)`,
  `parse_session_open` → generate own ephemeral → `derive_session_key` (salt = initiator's
  `bootstrap_nonce`) → write `session_keys(my_role=RESPONDER)` → `store.create_session` → return
  the leg-2 reply `session_open` to push at the now-persisted initiator mailbox.
- `complete_session(store, identity, peer, reply_envelope, now)` — initiator: load pending
  ephemeral → derive → write `session_keys(my_role=INITIATOR)` → `store.create_session` → clear
  pending.
Roundtrip test: both sides derive the **same** key (KAT-style); key + sessions row survive a
store reload; a mid-handshake restart reloads the pending row. Criteria: **B0-1, B0-2, B0-3** +
NU1 (session_id binding), NU2 (`create_session` wired), NF3 (shared bootstrap_nonce). *(Blocks
T3/T4.)*

**T3 — Outbound transport `send_fn` (`transport.py`).** Build the encrypt+sign+push closure from
`build_envelope` + `RelayClient.push`, keyed by `session_key`/`my_role`/sequence loaded from the
T2.5 `session_keys` table. Round-trip test against a second identity's `verify_and_open`.
Criteria: B1. *(Blocked by T2.5.)*

**T4 — Inbound adapter + flood cap (`transport.py`).** `fetch → group by sender_key_id (F3) →
(clamp each peer-group to INBOUND_FLOOD_CAP) → load stored session_key (drop+log
inbound_no_session if absent, B0-3) → verify_and_open (§8 order, peer_role, seq high-water) →
decrypt → append_transcript → set high-water → ack only processed`. A `session_open` control
envelope routes to T2.5 (`on_session_open` if no pending row for it, else `complete_session`),
not the message path; the reply leg is pushed. Replay/tamper rejected,
logged, no transcript. Criteria: B2, B3, C1, C2, B0-2, B0-3. *(Blocked by T2.5.)*

**T5 — Persisted cadence (extend existing helper, M2).** Reuse/extend the wall-clock cadence
helpers already in `relationship.py:211-242` (or factor them into one shared kindled-link cadence
helper) for the tick; add relay-error backoff. Do **not** clone `brain/soul/cadence.py`. Criteria:
supports A3, relay-health.

**T6 — The tick (`tick.py`) + through-path.** `_run_kindled_link_tick(persona_dir, store,
provider, config, now)`: cadence gate → recover() → for each paired peer: inbound poll+ingest
(T4) → schedule a gated response via `SessionEngine.process_outbound` with the real
`transcript_summary` → autonomous start only if `config.kindled_link_enabled` AND
`can_start_session`. Fault-isolated. Criteria: **A1, A2, A3, D1, D4, E4** (the §14 through-path
test lives here — drive the live tick).

**T7 — Supervisor wiring.** `run_folded(... kindled_link_enabled=True ...)` coarse param +
`_maybe_run_kindled_link_tick` block (store-owning ExitStack, fault-isolated), mirroring the
notes block at `supervisor.py:496–505`. The **real** user gate is `config.kindled_link_enabled`
read inside the tick (T6), exactly as notes reads `config.notes_enabled`. Criteria: A3 on the
true supervisor path.

**T8 — Cap atomicity + single clock.** Make the cap increment **atomic at the store level**
(`UPDATE peer_counters SET …=…+1 … RETURNING` so the post-increment value is read in one
statement) — **not** a lock-by-convention: the v0.0.37 double-bridge case means two processes can
touch the same SQLite file, so a documented single-writer-tick is insufficient (L3). Resolve
`_check_day` to one authoritative clock at the tick boundary. Criteria: F1; closes the Phase-4
atomicity carry + Phase-4/5 `_check_day` carry.

**T9 — Recovery surface.** `recover()` passes the real `transcript_summary`; a recovered-state
flag is exposed on a view/status field; panel recovery banner. Criteria: G1, G2.

**T10 — Gate `familiar` tier + stage-blind re-assert.** Add a mild `familiar` self-disclosure
latitude tier in `_build_gate_prompt`; re-run the stage-blind user-bar property across all
stages. Criteria: H1, E1 (carried).

**T11 — Relationship gradual-regression signal + rejection logging.** Add a **new external-signal
parameter** to `run_relationship_reflection` (F4 — the seam does not exist today; regression is
LLM-output-only) carrying hold-frequency / emotion-pressure computed by the tick from
`transport.jsonl`/holds; it can drive a gradual −1. Log ungrounded reflection quotes (audit-tier).
Criteria: H3, H4.

**T12 — Lost-path provenance + leak re-assert + relay-health/redacted-hold-reason views.**
`_peer_attributed` on the graveyard render path — **first confirm the graveyard entry actually
carries `memory_type`/a peer-marker** through `search_with_loss` (M3); if it doesn't, thread it
through so attribution is real, not a stubbed-test pass. Extend the Phase-6 held-body leak test
across the new fields; relay-health + redacted hold-reason on `views.py`/panel. Criteria: H2, E3,
relay-health + redacted hold-reason carries.

**T13 — Self-hostable relay + wrap-up.** `relay/dev_relay.py` host/port via env/CLI + `__main__`
+ run-note. Maturity-manifest entry (kindled-link → EXPERIMENTAL-on-live). Move 7b defers into
`project_companion_emergence_deferred.md` + next brainstorm. Criteria: I1; deferred-ledger rule.

## Build discipline

- Fixed `datetime`/`today` passed in every test (no internal clock; the project Date.now ban).
- Every peer provider call stays under `cli_throttle.background_slot` (E4) — do not regress to a
  bare `provider.complete`.
- `_send_allowed` runs FIRST in any send path (the Phase-3..5 ordering invariant).
- After build: `graphify update .`; full `uv run pytest -p no:randomly` + ruff + `pnpm test` +
  **`pnpm build`**.

## Self-review (plan author) — revised after stage-3 bounce #1

- **Spec coverage:** §3.0 (session layer, the blocker fix)→T2.5; §3.1→T6/T7; §3.2→T3/T4;
  §3.3→T2/T4; §3.4→T1; §3.5→T9; §3.6→T13; §3.7→T6; §3.8 carries → T8 (atomicity, clock), T9
  (transcript_summary), T11 (regression, rejection log), T10 (familiar tier), T12 (lost
  provenance, relay-health, redacted hold). No §3 item unmapped.
- **Criteria coverage:** B0→T2.5; A→T6; B→T3/T4; C→T2/T4; D→T1/T6; E→T6/T10/T12; F→T8; G→T9;
  H→T10/T11/T12; I→T13. Every criterion has a task.
- **Stage-3 findings dispositions (carried in `decisions.md`):** F1/L2/U1 (blocker)→§3.0+T2.5
  (session layer + `session_keys` table); F2→T10 + criteria H1 reworded (gate `:78` tuple, not
  `_STAGE_GUIDANCE`); F3→T4 group-by-`sender_key_id`; F4→T11 new reflection param; L1→§3.1 +T7
  (param True / config False); L3→T8 atomic `RETURNING`; M1→T2.5 table; M2→T5 reuse cadence;
  M3→T12 graveyard `memory_type`; U2→§3.3 logged residual; U4→§3.6 loopback/LAN bind; U3 (advisory
  metric may be empty)→stage-8 one-line check. All addressed in the revision.
- **Task dependency:** **T2.4 (addressing) → T2.5 (session layer) → T3/T4 (transport)** — build
  in that order; transport needs a stored key, the key handshake needs addressable mailboxes.
- **Lap-2/lap-3 dispositions (decisions.md):** 2-message handshake (lap-2 NF1)→T2.5; mailbox
  addressing (lap-3 F-BLOCKER, decoupled-mailbox per Hana)→T2.4; session_id-clobber (lap-3
  F-MAJOR)→T2.5 open_session/on_session_open guards + B0-5; bidirectional non-literal test oracle
  (lap-3 F-MAJOR)→A1 + B0-4. Crypto/role layer confirmed correct in lap 3.
- **Invariant-preservation tasks are explicit** (E1/E2/E3 re-asserted on the live path, not
  assumed) — the activation's chief risk is silently breaking a Phase-4..6 spine, so each is a
  named test, not a hope.
- **Known open question for the red-team:** does any second background consumer besides this tick
  touch per-peer cap counters (if not, single-writer-tick suffices for F1; if so, T8 needs a real
  store-level lock)? Flagged for stage-3 to confirm against the source.
