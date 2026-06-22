# Gate log — kindled-link-phase-7 (append-only)

- **Gate 4 (plan red-team), worst severity = BLOCKER, route → stage 1 (spec).**
  Finding class: F1/L2/U1 — session establishment + session-key persistence missing. The
  built brain has no caller of the X25519 handshake (`derive_session_key`/`build_session_open`/
  `parse_session_open` are test-only) and the store has no session-key/role column, so live
  inbound decryption (A1) and replay-across-restart (B2) cannot connect. Spot-verified by main
  thread (handshake-callers grep, store schema, dev_relay.fetch). Carried-forward findings for
  the stage-1/2 revision: F2 (`_STAGE_GUIDANCE` mis-named — gate latitude is `privacy_gate.py:78`
  tuple, add `familiar`), F3 (flood cap must group by `sender_key_id`), F4 (regression-signal
  needs a new reflection param), L1 (param=True / config=False wording), L3+M1 (atomic
  increment-and-return; add `session_keys` table), M2 (reuse existing cadence helper), M3
  (graveyard entry must carry `memory_type`), U2 (cumulative re-verify), U4 (loopback/trusted-LAN
  bind only). Bounce #1 at gate 4. **Stopped for human: blocker restarts the loop — confirming
  direction before re-spec (scope grows by the session layer).** → Hana chose "fold into 7a,
  re-spec one cut."

- **Gate 4 re-review (bounce #2), worst severity = BLOCKER, route → stage 1.** Finding class:
  **same as bounce #1 (session establishment, gate 4 / §3.0).** The added session layer used a
  crypto-impossible 1-message handshake; `derive_session_key` needs the peer's ephemeral, so it is
  a mutual 2-message exchange (NF1/NF2/NL2). Plus NU1 (session_id binding) + NU2 (must call
  `create_session` or every response holds — dead organ). All bounce-#1 findings (F2-F4, L1, L3,
  M1, M2, M3, U2, U4) confirmed addressed. Spot-verified the handshake math (`protocol.py:66-78`).
  **ITERATION CAP REACHED: 2 bounces at gate 4 on the session-establishment class → stop, human
  breaks the tie.** These were *convergent* (each a distinct, verified correctness catch), not
  livelock-by-rephrasing — the loop is working. Fix shape now crypto-exact (NM1b:
  `pending_handshakes` table + 2-leg handshake + `create_session` + session_id binding). Surfaced
  to Hana with the corrected design.

- **Gate 4 lap-3 (Hana elected one more lap), worst severity = BLOCKER, route → stage 1.**
  Crypto/role layer CONFIRMED CORRECT (prior two blockers genuinely closed: roles+nonce-space,
  2-message topology, leg disambiguation all clean). **New blocker one layer down: peer→mailbox
  addressing is undefined** — `create_invite`/`peers`/`session_open` carry no mailbox, so neither
  handshake leg can be addressed on a live relay. Spot-verified (`pairing.py:30-37`, `store.py:24-32`).
  Key fact: `peer_id == fingerprint == key_id` (key_id always known; only mailbox mapping missing).
  Plus F-MAJOR session_id-clobber guard + F-MAJOR bidirectional-test-oracle. This is a genuine
  ARCHITECTURE/PRIVACY decision (mailbox==key_id vs decoupled mailbox), not auto-fixable →
  surfaced to Hana. Three laps, three distinct convergent blockers — the loop de-risked a layer
  Phases 1-6 never specified (peer addressing).

- **Gate 4 resolution (Hana tie-break).** Addressing decision = **decoupled mailbox**
  (privacy-correct, matches parent §6.2): own mailbox random+persisted+decoupled-from-key_id;
  invite carries inviter mailbox → new `peers.relay_mailbox`; `build_session_open` carries signed
  `sender_mailbox`. Folded into §3.0 + T2.4 + B0-4/B0-5; clobber guard + bidirectional oracle
  folded in. **Hana routed FORWARD to build** (not a 4th plan lap) — crypto confirmed clean in
  lap 3; Stage 6 cold code red-team is the remaining independent challenge. Gate 4 → stage 5.

- **Gate 6 (code red-team), worst severity = MAJOR, route = fix-in-place → proceed.** Stage-6
  cold whole-branch reviewer (Opus): both load-bearing safety invariants HOLD (off-by-default;
  no-peer-content-flips-send); crypto/caps/flood/replay/provenance/tool-isolation all HOLD. Two
  MAJOR dead-readers (fail-safe, not leaks): recovered.flag had no writer (G2); live path never
  persisted a draft so recover/holds/H4 were empty + holds queried 'held' not 'hold'. Both FIXED
  (commit 52932052) + tests; AST oracle extended to tick/transport/session. Gate 7 → harness.
- **Gate 8 (harness), worst severity = clean → DONE.** Conformance PASS on all criteria (I2 met
  after the stage-6 fixes — no dead organ). Regression advisory + structurally clean
  (off-by-default ⇒ zero new standing load). Full gate: frontend 317 + pnpm build clean, ruff
  clean, backend full suite (appended at merge). → merge to main, EXPERIMENTAL, unreleased.
