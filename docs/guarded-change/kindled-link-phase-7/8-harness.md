# Stage 8 — Harness (Phase 7a)

Conformance (always) + regression (advisory, per Layer-2 config). The check command is the
project full gate: `uv run pytest -p no:randomly` + ruff + frontend `pnpm test` + `pnpm build`.

## Conformance — measured vs `1.5-criteria.md`

Every criterion is backed by a real, non-vacuous test (verified during the build per-task reviews
AND independently spot-checked by the stage-6 cold whole-branch reviewer, which read the
assertions). Result per group:

| Criterion | Verdict | Evidence |
|---|---|---|
| **B0-1..B0-5** session handshake (same key, restart, clobber, addressing, drop-no-session) | PASS | `test_session_handshake.py`, `test_store_session_keys.py`, `test_session_open_mailbox.py` |
| **A1/A2/A3** §14 through-path (transcript / gate+relationship read / live tick entry) | PASS | `test_tick.py::test_through_path_a1/a2`, drives `run_kindled_link_tick` |
| **B1/B2/B3** transport (round-trip / replay-after-restart / reject reasons) | PASS | `test_transport_ingest.py`, `test_transport_send.py` |
| **C1/C2** flood cap (per-peer bound, decrypt-work bound via call-count spy) | PASS | `test_transport_ingest.py` flood tests |
| **D1/D2/D3/D4** off-by-default (no outbound/start when disabled; default False; auth+strict endpoint; inbound still ingested) | PASS | `test_tick.py::test_d1/d4`, `test_kindled_link_config_endpoint.py` |
| **E1-E4** invariants (no-peer-flips-send; tool-less AST; no held-body leak; throttle slot) | PASS | `test_privacy_gate_adversarial.py`, `test_phase3_conformance.py` (now covers tick/transport/session), `test_views.py`+HTTP leak test, `test_tick.py::e4` |
| **F1** cap atomicity (single-statement reserve) | PASS | `test_store_atomic_reserve.py`, `test_engine_atomic_cap.py` |
| **G1/G2** recovery (real transcript_summary re-gate / recovered indicator set) | PASS | `test_session_engine_recovery.py`, `test_tick.py::test_recovered_flag_written` (stage-6 fix) |
| **H1** gate `familiar` tier + stage-blind user bar | PASS | `test_gate_stage.py` (5-stage byte-identity) |
| **H2** lost-path provenance | PASS | `test_kindled_peer_memory.py::...graveyard_dict_entry` |
| **H3/H4** rejection log + gradual-regression signal | PASS | `test_relationship_*` + `test_tick.py` reflection firing |
| **I1** full gate (pytest + ruff + pnpm test + pnpm build) | PASS | see below |
| **I2** `[rubric]` no dead organ / legible off-by-default | PASS (after stage-6 fixes) | stage-6 review confirmed both dead readers fed; off-by-default obvious in UI |

**I1 measured:** frontend `pnpm test` **317 passed**, `pnpm build` (tsc+vite) **clean**; ruff
**clean**; backend `uv run pytest -p no:randomly` — see the gate-run result appended at merge.

## Regression — ADVISORY ONLY (Layer-2 config: no comparable replay workload exists)

Per `guarded-change.companion.md`, regression metrics (`cost_per_chat_call_usd`,
`num_turns_per_chat_call`, `cache_*`) are **advisory** for this project until a fixed replay
workload exists. For Phase 7a the regression read is structurally clean: the feature is
**off-by-default**, so the standing persona exercises **zero** new chat/provider load — the
gating chat metrics are unchanged by construction (no new `call_type==chat` rows from kindled
work; peer provider calls are background `generate` rows that only occur when a user opts in AND
pairs a peer AND configures a relay). No regression to surface.

## Residual / out-of-scope (logged, not blocking)
- Real cross-host network validation, sustained-flood cumulative re-verify (U2), abuse/quota
  hardening, public-relay checklist → **7b** (spec §4).
- redacted-hold-reason category → **7b** (holds count surfaces).
- Carried minors: `has_fresh_inbound` coarse; gate/reflection `incr_provider_count` non-atomic
  (sequential in-tick); `_count_recent_holds` reaches `store._conn`.

## Verdict
**Conformance: PASS (all criteria, I2 after stage-6 fixes). Regression: clean (advisory; off-by-
default ⇒ zero new standing load).** → proceed to merge.
